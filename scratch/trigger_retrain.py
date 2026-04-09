"""Trigger a one-shot retrain with improved settings and print result."""
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

from quant_v2.research.scheduled_retrain import retrain_and_promote

model_root    = Path(os.getenv("BOT_MODEL_ROOT", "/app/models/production"))
registry_root = Path(os.getenv("BOT_MODEL_REGISTRY_ROOT", str(model_root / "registry")))

print("=" * 60)
print("Starting improved retrain: 12 months, BTCUSDT+ETHUSDT+BNBUSDT, min_accuracy=0.525")
print("=" * 60)

version_id = retrain_and_promote(
    model_root=model_root,
    registry_root=registry_root,
    train_months=12,
    min_accuracy=0.525,
    extra_symbols=["ETHUSDT", "BNBUSDT"],
)

if version_id:
    print(f"\nSUCCESS: Promoted {version_id} as active model.")
    print("Signal manager will hot-swap on next cycle (~1h).")
else:
    print("\nFAILED: No model passed the 0.525 accuracy gate.")
    print("Active model remains: model_20260220_215919 (Feb 20 rollback).")
    print("The Feb 20 model WILL produce trades — it has a 4m horizon ensemble.")
