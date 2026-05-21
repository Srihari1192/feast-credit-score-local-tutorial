# Run this script once after `feast apply` to register 15 SavedDatasets.
# Usage (from repo root): python create_datasets.py

from feast import FeatureStore
from feast.saved_dataset import SavedDataset, SavedDatasetStorage

store = FeatureStore(repo_path="feature_repo")

_FV_STRIDE = 13  # 15 * 13 = 195 — spreads 15 datasets across the 200 FVs

for i in range(15):
    fv_idx = (i * _FV_STRIDE) % 200
    entity_idx = fv_idx % 100

    dataset = SavedDataset(
        name=f"dataset_{i}",
        features=[f"fv_{fv_idx}:feature_{fv_idx}_val"],
        join_keys=[f"entity_{entity_idx}"],
        storage=SavedDatasetStorage(),
        full_feature_names=False,
        tags={"version": str(i), "purpose": "scale_test", "fv_ref": f"fv_{fv_idx}"},
    )
    store._registry.apply_saved_dataset(dataset, project=store.project, commit=True)
    print(f"Registered dataset_{i}  (fv_{fv_idx}, entity_{entity_idx})")

print("\nDone — 15 datasets registered in registry.")
