"""
Streamlit dashboard for COMP5339 Assignment 2.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import folium
import pandas as pd
import paho.mqtt.client as mqtt
import streamlit as st
import streamlit.components.v1 as components

from src.mqtt_helpers import BROKER, PORT, TOPIC, make_mqtt_client, merge_market, load_stream_csv


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
STREAM_PATH = PROCESSED_DIR / "latest_facility_stream_for_dashboard.csv"
MARKET_PATH = PROCESSED_DIR / "nem_market_price_demand_5min_may2026.csv"


st.set_page_config(
    page_title="NEM Facility Power and Emissions",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_cached_latest_data() -> pd.DataFrame:
    """Load the cached week and keep the latest record for every facility."""
    stream_df = merge_market(load_stream_csv(STREAM_PATH), MARKET_PATH)
    latest_df = (
        stream_df.sort_values(["timestamp", "facility_code"])
        .drop_duplicates(subset=["facility_code"], keep="last")
        .reset_index(drop=True)
    )
    return latest_df


def normalise_message_frame(messages: list[dict]) -> pd.DataFrame:
    if not messages:
        return pd.DataFrame()
    df = pd.DataFrame(messages)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ["power_mw", "emissions_t_co2e", "latitude", "longitude", "price_aud_per_mwh", "demand_mw"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return (
        df.dropna(subset=["timestamp", "facility_code", "latitude", "longitude"])
        .sort_values(["timestamp", "facility_code"])
        .drop_duplicates(subset=["facility_code"], keep="last")
        .reset_index(drop=True)
    )


@st.cache_resource(show_spinner=False)
def get_mqtt_store() -> dict:
    """Start one process-level MQTT subscriber and return its shared message store."""
    store = {
        "messages": [],
        "lock": threading.Lock(),
        "client": None,
        "receive_enabled": True,
        "connection_error": None,
    }

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(TOPIC, qos=1)

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            return
        with store["lock"]:
            if not store["receive_enabled"]:
                return
            store["messages"].append(payload)
            if len(store["messages"]) > 10000:
                store["messages"][:] = store["messages"][-10000:]

    client = make_mqtt_client(f"comp5339_streamlit_dashboard_{int(time.time())}")
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(BROKER, PORT, keepalive=60)
        client.loop_start()
        store["client"] = client
    except Exception as exc:
        store["connection_error"] = str(exc)
    return store

def get_dashboard_df(use_live_messages: bool, mqtt_store: dict) -> pd.DataFrame:
    cached_df = load_cached_latest_data()
    if not use_live_messages:
        return cached_df

    with mqtt_store["lock"]:
        messages = list(mqtt_store["messages"])
    live_df = normalise_message_frame(messages)
    if live_df.empty:
        return cached_df

    # Use cached data as a metadata backstop for facilities not yet observed live.
    metadata_cols = [
        "facility_code",
        "facility_name",
        "state",
        "network_region",
        "facility_type",
        "latitude",
        "longitude",
    ]
    metadata = cached_df[metadata_cols].drop_duplicates("facility_code")
    live_df = live_df.merge(metadata, on="facility_code", how="left", suffixes=("", "_cached"))
    for col in ["facility_name", "state", "network_region", "facility_type", "latitude", "longitude"]:
        cached_col = f"{col}_cached"
        if cached_col in live_df.columns:
            live_df[col] = live_df[col].fillna(live_df[cached_col])
            live_df = live_df.drop(columns=[cached_col])

    unseen = cached_df[~cached_df["facility_code"].isin(live_df["facility_code"])]
    return pd.concat([live_df, unseen], ignore_index=True, sort=False)


def build_facility_map(dataframe: pd.DataFrame, metric: str) -> folium.Map:
    df = dataframe.dropna(subset=["latitude", "longitude"]).copy()
    if df.empty:
        return folium.Map(location=[-25.5, 134.5], zoom_start=4, tiles="OpenStreetMap")

    metric_config = {
        "power_mw": {
            "radius_col": "power_mw",
            "color": "#2563eb",
            "label": "Power",
            "unit": "MW",
            "min_radius": 4,
            "max_extra": 20,
            "max_radius": 24,
        },
        "emissions_t_co2e": {
            "radius_col": "emissions_t_co2e",
            "color": "#dc2626",
            "label": "Emissions",
            "unit": "tCO2e",
            "min_radius": 3,
            "max_extra": 13,
            "max_radius": 16,
        },
    }[metric]

    radius_values = pd.to_numeric(df[metric_config["radius_col"]], errors="coerce").fillna(0).clip(lower=0)
    max_radius_value = max(float(radius_values.max()), 1.0)

    m = folium.Map(
        location=[df["latitude"].mean(), df["longitude"].mean()],
        zoom_start=5,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    for _, row in df.iterrows():
        radius_value = max(float(row.get(metric_config["radius_col"], 0) or 0), 0)
        radius = metric_config["min_radius"] + metric_config["max_extra"] * math.sqrt(
            radius_value / max_radius_value
        )
        radius = min(max(radius, metric_config["min_radius"]), metric_config["max_radius"])

        price = row.get("price_aud_per_mwh")
        demand = row.get("demand_mw")
        popup_html = f"""
        <div style="width: 310px">
            <h4 style="margin-bottom: 6px">{row.get('facility_name', 'Unknown')}</h4>
            <b>Code:</b> {row.get('facility_code', '')}<br>
            <b>Type:</b> {row.get('facility_type', '')}<br>
            <b>State/Region:</b> {row.get('state', '')} / {row.get('network_region', '')}<br>
            <b>Timestamp:</b> {row.get('timestamp', '')}<br>
            <b>Power:</b> {float(row.get('power_mw', 0) or 0):,.2f} MW<br>
            <b>Emissions:</b> {float(row.get('emissions_t_co2e', 0) or 0):,.2f} tCO2e<br>
            <b>NEM price:</b> {"N/A" if pd.isna(price) else f"{float(price):,.2f} $/MWh"}<br>
            <b>NEM demand:</b> {"N/A" if pd.isna(demand) else f"{float(demand):,.2f} MW"}
        </div>
        """

        tooltip = (
            f"{row.get('facility_name', 'Unknown')} | "
            f"{metric_config['label']}: {float(row.get(metric, 0) or 0):,.2f} {metric_config['unit']}"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color=metric_config["color"],
            fill=True,
            fill_color=metric_config["color"],
            fill_opacity=0.72,
            weight=1.4,
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=tooltip,
        ).add_to(m)

    return m


mqtt_store = get_mqtt_store()

st.title("NEM Facility Power and Emissions Dashboard")

with st.sidebar:
    st.header("Controls")
    receive_messages = st.toggle("Receive MQTT messages", value=True)
    use_live_messages = st.toggle("Use live MQTT messages", value=True)
    auto_refresh = st.toggle("Auto refresh", value=False)
    refresh_seconds = st.slider("Refresh interval seconds", min_value=3, max_value=30, value=5)

    metric_label = st.radio(
        "Map metric",
        options=["Power MW", "Emissions tCO2e"],
        index=0,
        horizontal=False,
    )
    metric = "power_mw" if metric_label == "Power MW" else "emissions_t_co2e"

    clear_live_messages = st.button("Clear live messages")

with mqtt_store["lock"]:
    mqtt_store["receive_enabled"] = receive_messages

if clear_live_messages:
    with mqtt_store["lock"]:
        mqtt_store["messages"].clear()
    st.success("Live MQTT message buffer cleared. The dashboard is now using cached fallback data until new messages arrive.")

dashboard_df = get_dashboard_df(use_live_messages, mqtt_store)

all_states = sorted(dashboard_df["state"].dropna().astype(str).unique().tolist())
all_types = sorted(dashboard_df["facility_type"].dropna().astype(str).unique().tolist())

with st.sidebar:
    selected_states = st.multiselect("States", all_states, default=all_states)
    selected_types = st.multiselect("Facility types", all_types, default=all_types)
    save_html = st.button("Save current map HTML")

filtered_df = dashboard_df.copy()
if selected_states:
    filtered_df = filtered_df[filtered_df["state"].astype(str).isin(selected_states)]
if selected_types:
    filtered_df = filtered_df[filtered_df["facility_type"].astype(str).isin(selected_types)]

latest_timestamp = filtered_df["timestamp"].max() if not filtered_df.empty else "No data"
latest_market = filtered_df.dropna(subset=["price_aud_per_mwh", "demand_mw"], how="all")
latest_price = latest_market["price_aud_per_mwh"].dropna().iloc[-1] if not latest_market.empty and latest_market["price_aud_per_mwh"].notna().any() else None
latest_demand = latest_market["demand_mw"].dropna().iloc[-1] if not latest_market.empty and latest_market["demand_mw"].notna().any() else None
with mqtt_store["lock"]:
    live_message_count = len(mqtt_store["messages"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Facilities shown", f"{filtered_df['facility_code'].nunique():,}")
col2.metric("Messages received", f"{live_message_count:,}")
col3.metric("Current NEM price", "N/A" if latest_price is None else f"{latest_price:,.2f} $/MWh")
col4.metric("Current NEM demand", "N/A" if latest_demand is None else f"{latest_demand:,.0f} MW")

receive_status = "receiving" if receive_messages else "paused"
display_mode = "live MQTT + cached fallback" if use_live_messages else "cached fallback only"
st.caption(
    f"Latest event time: {latest_timestamp} | MQTT receiving: {receive_status} | "
    f"Display mode: {display_mode} | Broker: {BROKER}:{PORT} | Topic: {TOPIC}"
)

if mqtt_store.get("connection_error"):
    st.warning(
        "MQTT live connection is unavailable, so the dashboard is showing cached data. "
        f"Connection error: {mqtt_store['connection_error']}"
    )

facility_map = build_facility_map(filtered_df, metric)

if save_html:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_name = "power" if metric == "power_mw" else "emissions"
    output_path = OUTPUT_DIR / f"streamlit_dashboard_{metric_name}_map.html"
    facility_map.save(output_path)
    st.success(f"Saved current map to {output_path}")

components.html(facility_map._repr_html_(), height=720, scrolling=False)

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
