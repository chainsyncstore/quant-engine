import yaml
from pathlib import Path


def test_execution_engine_service_removed():
    compose_path = Path(__file__).resolve().parents[2] / "docker-compose.yml"
    data = yaml.safe_load(compose_path.read_text())
    assert "execution_engine" not in data.get("services", {}), (
        "quant_execution service was removed in audit_20260423 P3-3; "
        "re-enable only after wiring Redis bus publishing in telegram bot."
    )
