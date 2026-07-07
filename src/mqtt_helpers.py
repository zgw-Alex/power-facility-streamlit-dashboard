"""MQTT publishing/subscribing helpers for Assignment 2 notebooks."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import paho.mqtt.client as mqtt


BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "comp5339/2026s1/team_xinyi/electricity/facilities"
PUBLISH_DELAY_SECONDS = 0.1
ROUND_DELAY_SECONDS = 60


def make_mqtt_client(client_id: str) -> mqtt.Client:
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=client_id)
    except Exception:
        return mqtt.Client(client_id=client_id)


def load_stream_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None)
    if "facility_code" not in df.columns and "facility_id" in df.columns:
        df["facility_code"] = df["facility_id"].astype(str)
    for col in ["network_region", "facility_type", "state", "facility_name"]:
        if col not in df.columns:
            df[col] = ""
    df = df.dropna(subset=["timestamp", "facility_code", "facility_name"]).copy()
    for col in ["power_mw", "emissions_t_co2e", "latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["timestamp", "facility_code"]).reset_index(drop=True)


def merge_market(stream_df: pd.DataFrame, market_path: str | Path) -> pd.DataFrame:
    if not Path(market_path).exists():
        stream_df["price_aud_per_mwh"] = pd.NA
        stream_df["demand_mw"] = pd.NA
        return stream_df
    market = pd.read_csv(market_path)
    market["timestamp"] = pd.to_datetime(market["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    return stream_df.merge(market, on="timestamp", how="left")


def row_to_message(row: pd.Series) -> dict[str, Any]:
    return {
        "timestamp": pd.to_datetime(row["timestamp"]).isoformat(),
        "facility_code": str(row["facility_code"]),
        "facility_name": str(row["facility_name"]),
        "state": str(row.get("state", "")),
        "network_region": str(row.get("network_region", "")),
        "facility_type": str(row.get("facility_type", "")),
        "latitude": float(row["latitude"]) if pd.notna(row.get("latitude")) else None,
        "longitude": float(row["longitude"]) if pd.notna(row.get("longitude")) else None,
        "power_mw": float(row.get("power_mw", 0) or 0),
        "emissions_t_co2e": float(row.get("emissions_t_co2e", 0) or 0),
        "price_aud_per_mwh": None if pd.isna(row.get("price_aud_per_mwh")) else float(row.get("price_aud_per_mwh")),
        "demand_mw": None if pd.isna(row.get("demand_mw")) else float(row.get("demand_mw")),
    }


def message_id(message: dict[str, Any]) -> str:
    identity = {
        "timestamp": message["timestamp"],
        "facility_code": message["facility_code"],
        "power_mw": round(float(message["power_mw"]), 6),
        "emissions_t_co2e": round(float(message["emissions_t_co2e"]), 6),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


def load_published_state(state_path: str | Path) -> set[str]:
    path = Path(state_path)
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")))


def save_published_state(state_path: str | Path, published: set[str]) -> None:
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(published), indent=2), encoding="utf-8")


def publish_dataframe(
    dataframe: pd.DataFrame,
    state_path: str | Path,
    broker: str = BROKER,
    port: int = PORT,
    topic: str = TOPIC,
    delay_seconds: float = PUBLISH_DELAY_SECONDS,
    only_new: bool = True,
    max_messages: int | None = None,
) -> int:
    publish_df = dataframe.sort_values(["timestamp", "facility_code"]).copy()
    if max_messages:
        publish_df = publish_df.head(max_messages)

    client = make_mqtt_client(f"comp5339_publisher_{int(time.time())}")
    client.connect(broker, port, keepalive=60)
    client.loop_start()

    published = load_published_state(state_path)
    sent_count = 0
    try:
        for _, row in publish_df.iterrows():
            message = row_to_message(row)
            mid = message_id(message)
            if only_new and mid in published:
                continue
            result = client.publish(topic, json.dumps(message), qos=1)
            result.wait_for_publish(timeout=10)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                raise RuntimeError(f"MQTT publish failed with code {result.rc}")
            published.add(mid)
            sent_count += 1
            time.sleep(delay_seconds)
    finally:
        save_published_state(state_path, published)
        client.loop_stop()
        client.disconnect()
    return sent_count


def run_continuous_cached_publisher(
    csv_path: str | Path,
    market_path: str | Path,
    state_path: str | Path,
    only_new: bool = True,
    round_delay_seconds: float = ROUND_DELAY_SECONDS,
) -> None:
    round_no = 1
    while True:
        stream_df = merge_market(load_stream_csv(csv_path), market_path)
        sent = publish_dataframe(stream_df, state_path=state_path, only_new=only_new)
        print(f"Round {round_no}: published {sent} new or changed records.")
        round_no += 1
        time.sleep(round_delay_seconds)
