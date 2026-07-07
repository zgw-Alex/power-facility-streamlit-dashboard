from __future__ import annotations

import argparse
import json
import os
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


BASE_URL = "https://api.openelectricity.org.au/v4"
NETWORK = "NEM"
DATE_START = "2026-05-01T00:00:00"
DATE_END = "2026-05-08T00:00:00"
INTERVAL = "5m"
REQUEST_TIMEOUT_SECONDS = 90
REQUEST_RETRIES = 4
REQUEST_BACKOFF_SECONDS = 3
DEFAULT_TOPIC_SAFE_GROUP = "team_xinyi"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "openelectricity"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
A1_PATH = PROJECT_ROOT / "data" / "raw" / "A1" / "final_dataset_for_storage.csv"


def load_env_file(env_path: Path | None = None) -> None:
    """Load simple KEY=VALUE pairs from .env without requiring extra packages."""
    path = env_path or PROJECT_ROOT / ".env"
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def api_key() -> str:
    load_env_file()
    key = os.environ.get("OPENELECTRICITY_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Set OPENELECTRICITY_API_KEY in a project-root .env file or environment variable. "
            "Example .env line: OPENELECTRICITY_API_KEY=your_key"
        )
    return key


def get_json(path: str, params: list[tuple[str, str]] | dict[str, str], key: str) -> dict[str, Any]:
    if isinstance(params, dict):
        query = urlencode(params)
    else:
        query = urlencode(params)
    url = f"{BASE_URL}{path}"
    if query:
        url = f"{url}?{query}"
    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "User-Agent": "COMP5339-Assignment2/1.0",
        },
    )
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenElectricity HTTP {exc.code} for {url}: {body[:500]}") from exc
        except (URLError, TimeoutError, ConnectionResetError, json.JSONDecodeError, OSError) as exc:
            if attempt == REQUEST_RETRIES:
                raise RuntimeError(
                    f"Could not read OpenElectricity response after {REQUEST_RETRIES} attempts for {url}: {exc}"
                ) from exc
            wait_seconds = REQUEST_BACKOFF_SECONDS * attempt
            print(
                f"OpenElectricity request failed on attempt {attempt}/{REQUEST_RETRIES}; "
                f"retrying in {wait_seconds}s ({type(exc).__name__}: {exc})"
            )
            time.sleep(wait_seconds)
    if not payload.get("success", False):
        raise RuntimeError(f"OpenElectricity returned unsuccessful response for {url}: {payload}")
    return payload


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace(".", "", regex=False)
    )
    return out


def clean_key(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def normalise_facility_name(value: Any) -> str:
    text = str(value).lower().strip()
    replacements = {
        "&": " and ",
        " power station": "",
        " wind farm": "",
        " solar farm": "",
        " solar plant": "",
        " battery energy storage system": "",
        " battery": "",
        " hydro power station": " hydro",
        " generation": "",
        " facility": "",
        " project": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    text = " ".join(part for part in text.split() if part not in {"stage", "the"})
    return text


def best_a1_match(row: pd.Series, candidates: pd.DataFrame) -> pd.Series:
    same_state = candidates[candidates["state_key"] == str(row.get("state_key", ""))]
    if same_state.empty:
        return pd.Series({"a1_facility_name": pd.NA, "a1_match_score": 0.0})

    name_key = row["facility_name_normalised"]
    exact = same_state[same_state["facility_name_normalised"] == name_key]
    if not exact.empty:
        match = exact.iloc[0]
        return pd.Series({"a1_facility_name": match["facility_name"], "a1_match_score": 1.0})

    best_score = 0.0
    best_row = None
    for _, candidate in same_state.iterrows():
        score = SequenceMatcher(None, name_key, candidate["facility_name_normalised"]).ratio()
        if score > best_score:
            best_score = score
            best_row = candidate
    if best_row is not None and best_score >= 0.72:
        return pd.Series({"a1_facility_name": best_row["facility_name"], "a1_match_score": best_score})
    return pd.Series({"a1_facility_name": pd.NA, "a1_match_score": best_score})


def flatten_facilities(facilities_json: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    facility_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []

    for facility in facilities_json.get("data", []):
        location = facility.get("location") or {}
        units = facility.get("units") or []
        fueltechs = sorted({str(unit.get("fueltech_id")) for unit in units if unit.get("fueltech_id")})
        capacity = sum(float(unit.get("capacity_registered") or 0) for unit in units)
        facility_rows.append(
            {
                "facility_code": facility.get("code"),
                "facility_name": facility.get("name"),
                "network_id": facility.get("network_id"),
                "network_region": facility.get("network_region"),
                "latitude_api": location.get("lat"),
                "longitude_api": location.get("lng"),
                "facility_type": ", ".join(fueltechs) if fueltechs else None,
                "registered_capacity_mw": capacity,
            }
        )
        for unit in units:
            unit_rows.append(
                {
                    "unit_code": unit.get("code"),
                    "unit_code_display": unit.get("code_display"),
                    "facility_code": facility.get("code"),
                    "facility_name": facility.get("name"),
                    "network_region": facility.get("network_region"),
                    "fueltech_id": unit.get("fueltech_id"),
                    "dispatch_type": unit.get("dispatch_type"),
                    "unit_status": unit.get("status_id"),
                    "capacity_registered": unit.get("capacity_registered"),
                }
            )

    facilities_df = pd.DataFrame(facility_rows).dropna(subset=["facility_code"])
    units_df = pd.DataFrame(unit_rows).dropna(subset=["unit_code", "facility_code"])
    return facilities_df, units_df


def fetch_facilities(key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = [("network_id", NETWORK), ("status_id", "operating")]
    payload = get_json("/facilities/", params, key)
    save_json(payload, RAW_DIR / "facilities_nem_operating_raw.json")
    return flatten_facilities(payload)


def fetch_facility_timeseries(
    key: str,
    facility_codes: list[str],
    date_start: str,
    date_end: str,
    interval: str,
    chunk_size: int,
    request_sleep: float,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    code_chunks = chunks(facility_codes, chunk_size)
    for index, code_chunk in enumerate(code_chunks, start=1):
        params: list[tuple[str, str]] = [
            ("metrics", "power"),
            ("metrics", "emissions"),
            ("interval", interval),
            ("date_start", date_start),
            ("date_end", date_end),
        ]
        params.extend(("facility_code", code) for code in code_chunk)
        payload = get_json(f"/data/facilities/{NETWORK}", params, key)
        payloads.append(payload)
        print(f"Fetched facility chunk {index}/{len(code_chunks)} ({len(code_chunk)} facilities)")
        time.sleep(request_sleep)
    save_json({"chunks": payloads}, RAW_DIR / "facility_power_emissions_raw.json")
    return payloads


def parse_timeseries_chunks(payloads: list[dict[str, Any]], units_df: pd.DataFrame) -> pd.DataFrame:
    unit_to_facility = units_df.set_index("unit_code")["facility_code"].to_dict()
    unit_to_fueltech = units_df.set_index("unit_code")["fueltech_id"].to_dict()
    records: list[dict[str, Any]] = []

    for payload in payloads:
        for block in payload.get("data", []):
            metric = block.get("metric")
            value_col = f"{metric}_mw" if metric == "power" else "emissions_t_co2e"
            for result in block.get("results", []):
                unit_code = (result.get("columns") or {}).get("unit_code")
                if not unit_code:
                    name = str(result.get("name", ""))
                    prefix = f"{metric}_"
                    unit_code = name[len(prefix) :] if name.startswith(prefix) else name
                facility_code = unit_to_facility.get(unit_code)
                if not facility_code:
                    continue
                for item in result.get("data", []):
                    if isinstance(item, list) and len(item) >= 2:
                        records.append(
                            {
                                "timestamp": item[0],
                                "facility_code": facility_code,
                                "unit_code": unit_code,
                                "fueltech_id": unit_to_fueltech.get(unit_code),
                                value_col: item[1],
                            }
                        )

    long_df = pd.DataFrame(records)
    if long_df.empty:
        raise RuntimeError("No facility time-series rows were parsed from the API response.")

    long_df["timestamp"] = pd.to_datetime(long_df["timestamp"], errors="coerce")
    for col in ["power_mw", "emissions_t_co2e"]:
        if col not in long_df.columns:
            long_df[col] = 0.0
        long_df[col] = pd.to_numeric(long_df[col], errors="coerce").fillna(0.0)

    unit_level = (
        long_df.groupby(["timestamp", "facility_code", "unit_code", "fueltech_id"], dropna=False, as_index=False)[
            ["power_mw", "emissions_t_co2e"]
        ]
        .sum()
        .sort_values(["timestamp", "facility_code", "unit_code"])
    )
    unit_level.to_csv(PROCESSED_DIR / "unit_power_emissions_5min_may2026.csv", index=False)

    facility_level = (
        unit_level.groupby(["timestamp", "facility_code"], as_index=False)[["power_mw", "emissions_t_co2e"]]
        .sum()
        .sort_values(["timestamp", "facility_code"])
    )
    return facility_level


def add_facility_metadata(facility_level: pd.DataFrame, facilities_df: pd.DataFrame) -> pd.DataFrame:
    out = facility_level.merge(facilities_df, on="facility_code", how="left")
    out["state_api"] = out["network_region"].astype(str).str[:3].str.replace("1", "", regex=False)
    if A1_PATH.exists():
        a1 = standardise_columns(pd.read_csv(A1_PATH))
        if {"facility_name", "state", "latitude", "longitude"}.issubset(a1.columns):
            a1["state_key"] = a1["state"].astype(str).str.strip().str.upper()
            a1["facility_name_normalised"] = a1["facility_name"].map(normalise_facility_name)
            a1_keep = (
                a1.dropna(subset=["facility_name", "latitude", "longitude"])
                .drop_duplicates(subset=["facility_name_normalised", "state_key"])
                [
                    [
                        "facility_name",
                        "facility_name_normalised",
                        "state_key",
                        "latitude",
                        "longitude",
                        "reporting_entity",
                        "installed_capacity_mw",
                    ]
                ]
            )
            facility_match_base = out[["facility_code", "facility_name", "state_api"]].drop_duplicates().copy()
            facility_match_base["state_key"] = facility_match_base["state_api"]
            facility_match_base["facility_name_normalised"] = facility_match_base["facility_name"].map(
                normalise_facility_name
            )
            matches = facility_match_base.join(
                facility_match_base.apply(best_a1_match, axis=1, candidates=a1_keep)
            )
            matched_a1 = matches.merge(
                a1_keep,
                left_on=["a1_facility_name", "state_key"],
                right_on=["facility_name", "state_key"],
                how="left",
                suffixes=("", "_a1"),
            )
            matched_a1 = matched_a1.rename(
                columns={
                    "latitude": "latitude_a1",
                    "longitude": "longitude_a1",
                    "installed_capacity_mw": "installed_capacity_mw_a1",
                }
            )
            out = out.merge(
                matched_a1[
                    [
                        "facility_code",
                        "a1_facility_name",
                        "a1_match_score",
                        "latitude_a1",
                        "longitude_a1",
                        "reporting_entity",
                        "installed_capacity_mw_a1",
                    ]
                ],
                on="facility_code",
                how="left",
            )
        else:
            out["latitude_a1"] = pd.NA
            out["longitude_a1"] = pd.NA
            out["reporting_entity"] = pd.NA
            out["a1_facility_name"] = pd.NA
            out["a1_match_score"] = 0.0
            out["installed_capacity_mw_a1"] = pd.NA
    else:
        out["latitude_a1"] = pd.NA
        out["longitude_a1"] = pd.NA
        out["reporting_entity"] = pd.NA
        out["a1_facility_name"] = pd.NA
        out["a1_match_score"] = 0.0
        out["installed_capacity_mw_a1"] = pd.NA

    out["latitude"] = pd.to_numeric(out["latitude_a1"], errors="coerce").fillna(
        pd.to_numeric(out["latitude_api"], errors="coerce")
    )
    out["longitude"] = pd.to_numeric(out["longitude_a1"], errors="coerce").fillna(
        pd.to_numeric(out["longitude_api"], errors="coerce")
    )
    out["state"] = out["state_api"]

    ordered_cols = [
        "timestamp",
        "facility_code",
        "facility_name",
        "a1_facility_name",
        "a1_match_score",
        "state",
        "network_region",
        "facility_type",
        "latitude",
        "longitude",
        "latitude_api",
        "longitude_api",
        "latitude_a1",
        "longitude_a1",
        "power_mw",
        "emissions_t_co2e",
        "registered_capacity_mw",
        "installed_capacity_mw_a1",
        "reporting_entity",
    ]
    return out[ordered_cols].dropna(subset=["timestamp", "facility_code", "facility_name"]).reset_index(drop=True)


def fetch_market_data(key: str, date_start: str, date_end: str, interval: str) -> pd.DataFrame:
    params = [
        ("metrics", "price"),
        ("metrics", "demand"),
        ("interval", interval),
        ("date_start", date_start),
        ("date_end", date_end),
        ("primary_grouping", "network"),
    ]
    payload = get_json(f"/market/network/{NETWORK}", params, key)
    save_json(payload, RAW_DIR / "nem_market_price_demand_raw.json")
    frames: list[pd.DataFrame] = []
    for block in payload.get("data", []):
        metric = block.get("metric")
        col = "price_aud_per_mwh" if metric == "price" else "demand_mw"
        rows: list[dict[str, Any]] = []
        for result in block.get("results", []):
            for item in result.get("data", []):
                if isinstance(item, list) and len(item) >= 2:
                    rows.append({"timestamp": item[0], col: item[1]})
        if rows:
            frame = pd.DataFrame(rows)
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frames.append(frame.groupby("timestamp", as_index=False)[col].mean())
    if not frames:
        return pd.DataFrame(columns=["timestamp", "price_aud_per_mwh", "demand_mw"])
    market = frames[0]
    for frame in frames[1:]:
        market = market.merge(frame, on="timestamp", how="outer")
    return market.sort_values("timestamp").reset_index(drop=True)


def run_once(args: argparse.Namespace) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    key = api_key()

    facilities_df, units_df = fetch_facilities(key)
    facilities_df.to_csv(PROCESSED_DIR / "nem_facilities_metadata.csv", index=False)
    units_df.to_csv(PROCESSED_DIR / "nem_units_metadata.csv", index=False)

    facility_codes = sorted(facilities_df["facility_code"].dropna().unique().tolist())
    if args.max_facilities:
        facility_codes = facility_codes[: args.max_facilities]
    print(f"Retrieving {len(facility_codes)} facilities from {args.date_start} to {args.date_end}")

    payloads = fetch_facility_timeseries(
        key,
        facility_codes,
        args.date_start,
        args.date_end,
        args.interval,
        args.chunk_size,
        args.request_sleep,
    )
    facility_level = parse_timeseries_chunks(payloads, units_df)
    integrated = add_facility_metadata(facility_level, facilities_df)
    integrated = integrated.sort_values(["timestamp", "facility_code"]).reset_index(drop=True)
    out_path = PROCESSED_DIR / "facility_power_emissions_5min_may2026.csv"
    integrated.to_csv(out_path, index=False)
    print(f"Saved facility CSV: {out_path} rows={len(integrated):,}")

    market = fetch_market_data(key, args.date_start, args.date_end, args.interval)
    market_path = PROCESSED_DIR / "nem_market_price_demand_5min_may2026.csv"
    market.to_csv(market_path, index=False)
    print(f"Saved market CSV: {market_path} rows={len(market):,}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenElectricity Assignment 2 data retrieval and caching")
    parser.add_argument("--date-start", default=DATE_START)
    parser.add_argument("--date-end", default=DATE_END)
    parser.add_argument("--interval", default=INTERVAL)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--request-sleep", type=float, default=1.0)
    parser.add_argument("--max-facilities", type=int, default=0, help="Debug only; 0 means all facilities")
    return parser


if __name__ == "__main__":
    run_once(build_parser().parse_args())
