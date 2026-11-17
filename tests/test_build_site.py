import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_static_site_bundles_engine_without_server_modules(tmp_path):
    build = runpy.run_path(str(ROOT / "scripts" / "build_site.py"))["build"]
    build(tmp_path, "https://cdn.example/pyodide/")
    html = (tmp_path / "index.html").read_text()
    assert 'src="./boot.js"' in html and 'href="./style.css"' in html and "/static/" not in html
    assert '"https://cdn.example/pyodide/"' in (tmp_path / "boot.js").read_text()
    bundle = json.loads((tmp_path / "engine_bundle.json").read_text())
    assert "engine/bridge.py" in bundle and "engine/api.py" not in bundle
    assert any(k.startswith("examples/clinic_intake/") for k in bundle)
