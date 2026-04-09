"""Full production retrain with Phase 1+3 improvements.

12 months, 3 symbols (BTC+ETH+BNB), dead-zone labels, tuned hyperparameters.
"""
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

from quant_v2.research.scheduled_retrain import retrain_and_promote

model_root = Path(os.getenv("BOT_MODEL_ROOT", "/app/models/production"))
registry_root = Path(os.getenv("BOT_MODEL_REGISTRY_ROOT", str(model_root / "registry")))

print("=" * 60)
print("PRODUCTION: Phase 1+3 Full Retrain")
print("=" * 60)
print("12 months, BTC+ETH+BNB, min_accuracy=0.525")

version_id = retrain_and_promote(
    model_root=model_root,
    registry_root=registry_root,
    train_months=12,
    min_accuracy=0.525,
    extra_symbols=["ETHUSDT", "BNBUSDT"],
)

if version_id:
    print(f"\n✅ PRODUCTION MODEL PROMOTED: {version_id}")
else:
    print("\n❌ Failed - keeping previous active model")
