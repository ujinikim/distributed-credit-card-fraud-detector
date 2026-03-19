"""Synthetic transaction generator: Gaussian normal behavior + anomaly injection."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

# Default config keys (overridden by config/synthetic.yaml)
DEFAULT_NUM_TRANSACTIONS = 10_000
DEFAULT_ANOMALY_RATIO = 0.01
DEFAULT_IMPOSSIBLE_TRAVEL_FRACTION = 0.5
EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two (lat, lon) points."""
    a = math.radians(lat1)
    b = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlat / 2) ** 2 + math.cos(a) * math.cos(b) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, x)))


def _random_point_far_from(rng: random.Random, from_lat: float, from_lon: float, min_km: float) -> tuple[float, float]:
    """Return (lat, lon) at least min_km from (from_lat, from_lon)."""
    for _ in range(100):
        # Sample a point: offset in degrees (rough: 1 deg lat ~ 111 km)
        dlat = rng.gauss(0, 10)
        dlon = rng.gauss(0, 10)
        lat = max(-90, min(90, from_lat + dlat))
        lon = max(-180, min(180, from_lon + dlon))
        if _haversine_km(from_lat, from_lon, lat, lon) >= min_km:
            return (round(lat, 4), round(lon, 4))
    # Fallback: antipodal-ish
    return (max(-90, min(90, -from_lat)), max(-180, min(180, from_lon + 180)))


@dataclass
class GeneratorConfig:
    """Config for the synthetic generator (from config/synthetic.yaml)."""

    num_transactions: int
    start_date: str
    end_date: str
    seed: int
    anomaly_ratio: float
    num_cards: int
    amount_mean: float
    amount_std: float
    output_path: str
    merchant_categories: list[dict[str, Any]]
    geo_center_lat: float
    geo_center_lon: float
    geo_std: float
    min_minutes_between_locations: int
    min_distance_km_impossible_travel: float
    spike_amount_multiplier: float
    impossible_travel_fraction: float
    raw_write_mode: str
    normal_min_minutes_between_transactions: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GeneratorConfig:
        """Build config from a dict (e.g. from YAML)."""
        cats = d.get("merchant_categories") or [
            {"name": "retail", "weight": 0.40},
            {"name": "food", "weight": 0.25},
            {"name": "travel", "weight": 0.15},
            {"name": "other", "weight": 0.20},
        ]
        return cls(
            num_transactions=int(d.get("num_transactions", DEFAULT_NUM_TRANSACTIONS)),
            start_date=str(d.get("start_date", "2013-09-01T00:00:00")),
            end_date=str(d.get("end_date", "2013-09-30T23:59:59")),
            seed=int(d.get("seed", 42)),
            anomaly_ratio=float(d.get("anomaly_ratio", DEFAULT_ANOMALY_RATIO)),
            num_cards=int(d.get("num_cards", 500)),
            amount_mean=float(d.get("amount_mean", 75.0)),
            amount_std=float(d.get("amount_std", 40.0)),
            output_path=str(d.get("output_path", "data/raw")),
            merchant_categories=cats,
            geo_center_lat=float(d.get("geo_center_lat", 40.0)),
            geo_center_lon=float(d.get("geo_center_lon", -74.0)),
            geo_std=float(d.get("geo_std", 0.5)),
            min_minutes_between_locations=int(d.get("min_minutes_between_locations", 120)),
            min_distance_km_impossible_travel=float(d.get("min_distance_km_impossible_travel", 500.0)),
            spike_amount_multiplier=float(d.get("spike_amount_multiplier", 5.0)),
            impossible_travel_fraction=float(d.get("impossible_travel_fraction", DEFAULT_IMPOSSIBLE_TRAVEL_FRACTION)),
            raw_write_mode=str(d.get("raw_write_mode", "overwrite")),
            normal_min_minutes_between_transactions=int(d.get("normal_min_minutes_between_transactions", 30)),
        )


def _project_root() -> Path:
    """Project root (repo root): parent of src/."""
    return Path(__file__).resolve().parents[3]


def load_config(config_path: str | Path | None = None, paths_yaml: str | Path | None = None) -> GeneratorConfig:
    """Load generator config from synthetic.yaml and optionally override output_path from paths.yaml."""
    root = _project_root()
    if config_path is None:
        config_path = root / "config" / "synthetic.yaml"
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if paths_yaml is None:
        paths_yaml = root / "config" / "paths.yaml"
    paths_path = Path(paths_yaml)
    if paths_path.exists():
        with open(paths_path, encoding="utf-8") as f:
            paths = yaml.safe_load(f) or {}
        raw = paths.get("data", {}).get("raw")
        if raw:
            data["output_path"] = raw
    return GeneratorConfig.from_dict(data)


@dataclass
class Transaction:
    """Single transaction record (schema matches Bronze/Silver)."""

    transaction_id: str
    card_id: str
    event_time: str
    amount: float
    merchant_category: str
    latitude: float
    longitude: float
    anomaly_type: str
    ref_transaction_id: str | None = None  # for impossible_travel: the tx we "jumped" from

    def to_json_line(self) -> str:
        """One line of JSON (no trailing newline in dict; caller adds newline)."""
        out: dict[str, Any] = {
            "transaction_id": self.transaction_id,
            "card_id": self.card_id,
            "event_time": self.event_time,
            "amount": round(self.amount, 2),
            "merchant_category": self.merchant_category,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "anomaly_type": self.anomaly_type,
        }
        if self.ref_transaction_id is not None:
            out["ref_transaction_id"] = self.ref_transaction_id
        return json.dumps(out)


def _parse_iso(s: str) -> datetime:
    """Parse ISO-8601 datetime string to datetime."""
    if "T" in s:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    return datetime.fromisoformat(s + "T00:00:00+00:00")


def _format_iso(dt: datetime) -> str:
    """Format datetime to ISO-8601 string."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + "Z"


def _enforce_min_gap_by_card(
    transactions: list[Transaction], min_gap_minutes: int
) -> None:
    """Shift normal per-card events forward so speed features stay realistic for non-travel rows."""
    if min_gap_minutes <= 0:
        return

    min_gap = timedelta(minutes=min_gap_minutes)
    by_card: dict[str, list[Transaction]] = {}
    for tx in transactions:
        by_card.setdefault(tx.card_id, []).append(tx)

    for card_transactions in by_card.values():
        card_transactions.sort(key=lambda tx: (tx.event_time, tx.transaction_id))
        prev_dt: datetime | None = None
        for tx in card_transactions:
            tx_dt = _parse_iso(tx.event_time)
            if prev_dt is not None and tx_dt < prev_dt + min_gap:
                tx_dt = prev_dt + min_gap
                tx.event_time = _format_iso(tx_dt)
            prev_dt = tx_dt


def generate(config: GeneratorConfig, run_id: str | None = None) -> list[Transaction]:
    """
    Generate synthetic transactions: normal (Gaussian) + spending_spike and impossible_travel anomalies.
    Returns list of transactions (caller writes to JSONL).
    """
    rng = random.Random(config.seed)
    run_id = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    start = _parse_iso(config.start_date)
    end = _parse_iso(config.end_date)
    span_seconds = max(1, (end - start).total_seconds())

    # Build weighted merchant categories
    names: list[str] = []
    weights: list[float] = []
    for c in config.merchant_categories:
        names.append(c["name"])
        weights.append(float(c["weight"]))
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    # Per-card home region (Gaussian around global center)
    card_homes: dict[str, tuple[float, float]] = {}
    for i in range(config.num_cards):
        cid = f"card_{i:04d}"
        lat = max(-90, min(90, rng.gauss(config.geo_center_lat, config.geo_std)))
        lon = max(-180, min(180, rng.gauss(config.geo_center_lon, config.geo_std)))
        card_homes[cid] = (round(lat, 4), round(lon, 4))

    card_ids = list(card_homes.keys())
    n = config.num_transactions
    num_spike = int(n * config.anomaly_ratio * (1 - config.impossible_travel_fraction))
    num_travel = int(n * config.anomaly_ratio * config.impossible_travel_fraction)

    transactions: list[Transaction] = []
    tx_counter = 0

    def next_id() -> str:
        nonlocal tx_counter
        tx_counter += 1
        return f"tx_{run_id}_{tx_counter}"

    for i in range(n):
        card_id = rng.choice(card_ids)
        # Event time: uniform over range (optionally skew business hours; here uniform)
        t = start + timedelta(seconds=rng.uniform(0, span_seconds))
        event_time = _format_iso(t)
        amount = max(0.0, rng.gauss(config.amount_mean, config.amount_std))
        cat = rng.choices(names, weights=weights, k=1)[0]
        home_lat, home_lon = card_homes[card_id]
        lat = max(-90, min(90, rng.gauss(home_lat, config.geo_std)))
        lon = max(-180, min(180, rng.gauss(home_lon, config.geo_std)))
        lat, lon = round(lat, 4), round(lon, 4)

        anomaly_type = "none"
        if num_spike > 0 and rng.random() < (num_spike / max(1, n)):
            num_spike -= 1
            amount = max(0.0, config.amount_mean + config.spike_amount_multiplier * config.amount_std)
            anomaly_type = "spending_spike"

        transactions.append(
            Transaction(
                transaction_id=next_id(),
                card_id=card_id,
                event_time=event_time,
                amount=amount,
                merchant_category=cat,
                latitude=lat,
                longitude=lon,
                anomaly_type=anomaly_type,
            )
        )

    _enforce_min_gap_by_card(
        transactions, config.normal_min_minutes_between_transactions
    )

    # Impossible travel: add extra rows (same card, few minutes later, far location)
    by_card: dict[str, list[Transaction]] = {}
    for tx in transactions:
        by_card.setdefault(tx.card_id, []).append(tx)

    # For impossible_travel, always pair with the latest-known transaction for that
    # card so that, after we sort by event_time, the anomaly row is guaranteed to be
    # immediately after its reference in time for that card. This keeps the
    # Silver→Gold lag-based speed calculation aligned with the synthetic intent.
    candidates = [c for c in card_ids if len(by_card.get(c, [])) >= 1]
    for _ in range(num_travel):
        if not candidates:
            break
        card_id = rng.choice(candidates)
        recent = by_card[card_id]
        # Use the latest transaction for this card as the reference so that,
        # within that card's timeline, the impossible_travel row is always
        # adjacent and its speed_from_prev_kmh is computed vs this ref row.
        ref = max(recent, key=lambda t: _parse_iso(t.event_time))
        ref_dt = _parse_iso(ref.event_time)
        new_dt = ref_dt + timedelta(minutes=config.min_minutes_between_locations // 10)  # e.g. 12 min later
        far_lat, far_lon = _random_point_far_from(
            rng, ref.latitude, ref.longitude, config.min_distance_km_impossible_travel
        )
        amount = max(0.0, rng.gauss(config.amount_mean, config.amount_std))
        cat = rng.choices(names, weights=weights, k=1)[0]
        transactions.append(
            Transaction(
                transaction_id=next_id(),
                card_id=card_id,
                event_time=_format_iso(new_dt),
                amount=amount,
                merchant_category=cat,
                latitude=far_lat,
                longitude=far_lon,
                anomaly_type="impossible_travel",
                ref_transaction_id=ref.transaction_id,
            )
        )

    # Sort by event_time
    transactions.sort(key=lambda tx: tx.event_time)
    return transactions


def _clear_existing_jsonl(output_path: Path) -> None:
    """Remove prior JSONL outputs so each overwrite-mode run starts clean."""
    for existing in output_path.glob("*.jsonl"):
        existing.unlink()


def write_jsonl(
    transactions: list[Transaction],
    output_path: str | Path,
    run_id: str | None = None,
    write_mode: str = "overwrite",
) -> Path:
    """Write transactions to a JSONL file under output_path. Returns path to written file."""
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    run_id = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if write_mode == "overwrite":
        _clear_existing_jsonl(out)
        filepath = out / "transactions.jsonl"
    elif write_mode == "append":
        filepath = out / f"transactions_{run_id}.jsonl"
    else:
        raise ValueError(f"Unsupported raw_write_mode: {write_mode}")
    with open(filepath, "w", encoding="utf-8") as f:
        for tx in transactions:
            f.write(tx.to_json_line() + "\n")
    return filepath


def run(
    config_path: str | Path | None = None,
    paths_yaml: str | Path | None = None,
    run_id: str | None = None,
) -> Path:
    """
    Load config, generate transactions, write JSONL. Returns path to written file.
    """
    config = load_config(config_path=config_path, paths_yaml=paths_yaml)
    run_id = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    transactions = generate(config, run_id=run_id)
    return write_jsonl(
        transactions,
        config.output_path,
        run_id=run_id,
        write_mode=config.raw_write_mode,
    )
