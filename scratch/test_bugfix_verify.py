"""Verify bugfixes: split_idx, cfg, sample_weights, early stopping.

Quick 3-month, 2-symbol test to confirm no crashes.
"""
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

from quant_v2.research.scheduled_retrain import retrain_and_promote

model_root = Path(os.getenv("BOT_MODEL_ROOT", "/app/models/production"))
registry_root = Path(os.getenv("BOT_MODEL_REGISTRY_ROOT", str(model_root / "registry")))

print("=" * 60)
print("VERIFY: Bugfixes (split_idx, cfg, sample_weights, early stopping)")
print("=" * 60)

version_id = retrain_and_promote(
    model_root=model_root,
    registry_root=registry_root,
    train_months=3,
    min_accuracy=0.525,
    extra_symbols=["ETHUSDT"],
)

if version_id:
    print(f"\n✅ All phases working correctly. Promoted {version_id}")
else:
    print("\n❌ Model did not pass accuracy gate (check logs)")
