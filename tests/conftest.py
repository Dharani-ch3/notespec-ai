import pytest

from engine.config import Settings
from engine.core import Engine
from engine.store import Store


@pytest.fixture
def engine() -> Engine:
    return Engine(Settings(db_path=":memory:"), Store(":memory:"))


@pytest.fixture
def sample_project(engine):
    from pathlib import Path

    project = engine.create_project("Clinic Intake", "tester")
    for path in sorted((Path(__file__).parent.parent / "examples" / "clinic_intake").iterdir()):
        engine.ingest_bytes(project["id"], path.name, path.read_bytes(), "tester")
    return project
