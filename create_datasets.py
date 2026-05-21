# Run this script once after `feast apply` to register 15 SavedDatasets.
# Usage (from repo root): python create_datasets.py
#
# Writes directly to the registry so datasets appear in the Feast UI.

from feast import FeatureStore
from feast.saved_dataset import SavedDataset, SavedDatasetStorage

store = FeatureStore(repo_path="feature_repo")

_FV_STRIDE = 6  # 15 * 6 = 90 — spreads 15 datasets across the 100 FVs

for i in range(15):
    fv_idx = (i * _FV_STRIDE) % 100
    entity_idx = fv_idx % 20

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
