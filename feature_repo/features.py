# Scale test feature definitions
# 200 Entities | 60 Data Sources | 500 Feature Views | 500 Features | 120 Feature Services

from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.data_format import ParquetFormat
from feast.types import Float64, Int64, String
from feast.value_type import ValueType

# ── 200 Entities ─────────────────────────────────────────────────────────────
for _i in range(200):
    globals()[f"entity_{_i}"] = Entity(
        name=f"entity_{_i}",
        value_type=ValueType.INT64,
        description=f"Scale test entity {_i}",
    )

# ── 60 Data Sources ───────────────────────────────────────────────────────────
# Sources 0-1 reuse existing credit/zipcode parquet files; rest use generated paths.
_source_paths = [
    "data/credit_history.parquet",
    "data/zipcode_table.parquet",
    "data/loan_table.parquet",
] + [f"data/source_{_i}.parquet" for _i in range(3, 60)]

for _i in range(60):
    globals()[f"source_{_i}"] = FileSource(
        name=f"source_{_i}",
        path=_source_paths[_i],
        file_format=ParquetFormat(),
        timestamp_field="event_timestamp",
        created_timestamp_column="created_timestamp",
    )

# ── 500 Feature Views (1 field each = 500 Features) ──────────────────────────
_DTYPE_CYCLE = [Float64, Int64, String, Float64, Int64]

for _i in range(500):
    _entity = globals()[f"entity_{_i % 200}"]
    _source = globals()[f"source_{_i % 60}"]
    _dtype = _DTYPE_CYCLE[_i % len(_DTYPE_CYCLE)]
    globals()[f"fv_{_i}"] = FeatureView(
        name=f"fv_{_i}",
        entities=[_entity],
        ttl=timedelta(days=90 + (_i % 275)),
        schema=[
            Field(name=f"feature_{_i}_val", dtype=_dtype),
        ],
        source=_source,
        tags={
            "domain": f"domain_{_i % 10}",
            "tier": f"tier_{_i % 5}",
            "team": f"team_{_i % 8}",
        },
    )

# ── 120 Feature Services ──────────────────────────────────────────────────────
# Distribute 500 FVs across 120 services (4 FVs each, cycling through all 500).
for _i in range(120):
    _indices = [(_i * 4 + _j) % 500 for _j in range(4)]
    _fvs = [globals()[f"fv_{_idx}"] for _idx in _indices]
    globals()[f"feature_service_{_i}"] = FeatureService(
        name=f"feature_service_{_i}",
        features=_fvs,
        description=f"Scale test feature service {_i} — covers FVs {_indices}",
        tags={"owner": f"team_{_i % 8}", "sla": "standard"},
    )
