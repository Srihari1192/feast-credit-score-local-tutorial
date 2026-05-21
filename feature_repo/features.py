# Scale test feature definitions
# 20 Entities | 20 Data Sources | 100 Feature Views | 100 Features | 60 Feature Services

from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.data_format import ParquetFormat
from feast.types import Float64, Int64, String
from feast.value_type import ValueType

# ── 20 Entities ───────────────────────────────────────────────────────────────
for _i in range(20):
    globals()[f"entity_{_i}"] = Entity(
        name=f"entity_{_i}",
        value_type=ValueType.INT64,
        description=f"Scale test entity {_i}",
    )

# ── 20 Data Sources ───────────────────────────────────────────────────────────
_source_paths = [
    "data/credit_history.parquet",
    "data/zipcode_table.parquet",
    "data/loan_table.parquet",
] + [f"data/source_{_i}.parquet" for _i in range(3, 20)]

for _i in range(20):
    globals()[f"source_{_i}"] = FileSource(
        name=f"source_{_i}",
        path=_source_paths[_i],
        file_format=ParquetFormat(),
        timestamp_field="event_timestamp",
        created_timestamp_column="created_timestamp",
    )

# ── 100 Feature Views (1 field each = 100 Features) ───────────────────────────
_DTYPE_CYCLE = [Float64, Int64, String, Float64, Int64]

for _i in range(100):
    _entity = globals()[f"entity_{_i % 20}"]
    _source = globals()[f"source_{_i % 20}"]
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

# ── 60 Feature Services ───────────────────────────────────────────────────────
# 100 FVs across 60 services: first 40 services get 2 FVs, last 20 get 1 FV
# (40×2 + 20×1 = 100) — every FV in exactly one service.
_fv_cursor = 0
for _i in range(60):
    _count = 2 if _i < 40 else 1
    _indices = [(_fv_cursor + _j) % 100 for _j in range(_count)]
    _fv_cursor = (_fv_cursor + _count) % 100
    _fvs = [globals()[f"fv_{_idx}"] for _idx in _indices]
    globals()[f"feature_service_{_i}"] = FeatureService(
        name=f"feature_service_{_i}",
        features=_fvs,
        description=f"Scale test feature service {_i} — covers FVs {_indices}",
        tags={"owner": f"team_{_i % 8}", "sla": "standard"},
    )
