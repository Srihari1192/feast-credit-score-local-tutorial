# 15 SavedDataset definitions for scale testing.
# These are discovered by `feast apply` and registered in the feature registry.
# Each dataset references a feature view from features.py.

from feast.saved_dataset import SavedDataset, SavedDatasetFileStorage

# Stride across the 500 feature views so each dataset covers a different slice.
_FV_STRIDE = 33  # 15 * 33 = 495, covers 15 distinct regions of the 500 FVs

for _i in range(15):
    _fv_idx = (_i * _FV_STRIDE) % 500
    _entity_idx = _fv_idx % 200

    globals()[f"dataset_{_i}"] = SavedDataset(
        name=f"dataset_{_i}",
        features=[f"fv_{_fv_idx}:feature_{_fv_idx}_val"],
        join_keys=[f"entity_{_entity_idx}"],
        storage=SavedDatasetFileStorage(path=f"data/dataset_{_i}.parquet"),
        full_feature_names=False,
        tags={
            "version": str(_i),
            "purpose": "scale_test",
            "fv_ref": f"fv_{_fv_idx}",
        },
    )
