# Scale test feature definitions
# 100 Entities | 60 Data Sources | 200 Feature Views | 200 Features | 200 Feature Services

from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.data_format import ParquetFormat
from feast.types import Float64, Int64, String
from feast.value_type import ValueType

# ── 100 Entities ──────────────────────────────────────────────────────────────
for _i in range(100):
    globals()[f"entity_{_i}"] = Entity(
        name=f"entity_{_i}",
        value_type=ValueType.INT64,
        description=f"Scale test entity {_i}",
    )

# ── 60 Data Sources ───────────────────────────────────────────────────────────
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

# ── 200 Feature Views (1 field each = 200 Features) ───────────────────────────
_DTYPE_CYCLE = [Float64, Int64, String, Float64, Int64]

for _i in range(200):
    _entity = globals()[f"entity_{_i % 100}"]
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

# ── 200 Feature Services ──────────────────────────────────────────────────────
# 200 FVs across 200 services = exactly 1 FV per service — full coverage guaranteed.
for _i in range(200):
    globals()[f"feature_service_{_i}"] = FeatureService(
        name=f"feature_service_{_i}",
        features=[globals()[f"fv_{_i}"]],
        description=f"Scale test feature service {_i} — covers fv_{_i}",
        tags={"owner": f"team_{_i % 8}", "sla": "standard"},
    )
