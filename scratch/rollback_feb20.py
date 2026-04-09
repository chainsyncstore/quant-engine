"""One-shot rollback: set active model to model_20260220_215919 (Feb 20 - last known good)."""
from pathlib import Path
from quant_v2.model_registry import ModelRegistry

registry = ModelRegistry(Path("/app/models/production/registry"))

# First check what model files are available
feb20_dir = Path("/app/models/production/model_20260220_215919")
print(f"Feb 20 artifact dir exists: {feb20_dir.exists()}")
if feb20_dir.exists():
    print(f"Files: {[f.name for f in feb20_dir.iterdir()]}")

current = registry.get_active_pointer()
print(f"Current active: {current.version_id if current else 'None'}")

registry.set_active_version("model_20260220_215919")
new = registry.get_active_pointer()
print(f"New active    : {new.version_id if new else 'None'}")
print("Rollback complete. Signal manager will hot-swap on next cycle.")
