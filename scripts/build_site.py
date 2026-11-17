"""Build the static GitHub Pages demo into ./site.

The web console is reused as-is. Instead of a server, the page loads Pyodide
(CPython compiled to WebAssembly) and runs the engine's own Python code in the
visitor's browser, through engine/bridge.py.

    python scripts/build_site.py                      # Pyodide from the jsDelivr CDN
    python scripts/build_site.py --pyodide-url /pyodide/   # a self-hosted copy (offline testing)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "engine" / "static"
PYODIDE_CDN = "https://cdn.jsdelivr.net/pyodide/v0.29.5/full/"
# Server-only modules (FastAPI, MCP, CLI) are left out of the browser bundle.
SKIP = {"api.py", "mcp_server.py", "cli.py"}

BOOT_JS = """// Boots the Python engine in the browser with Pyodide.
window.KAE_STATIC = true;
(function () {
  const PYODIDE_URL = %(pyodide_url)s;
  // Look elements up when used: this script runs before the overlay below it is parsed.
  const say = (msg) => { const s = document.getElementById("bootStatus"); if (s) s.textContent = msg; };

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src;
      s.onload = resolve;
      s.onerror = () => reject(new Error("Could not load " + src));
      document.head.append(s);
    });
  }

  window.kaeReady = (async () => {
    try {
      say("Loading Python (WebAssembly)…");
      await loadScript(PYODIDE_URL + "pyodide.js");
      const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
      say("Loading pydantic and sqlite3…");
      await pyodide.loadPackage(["pydantic", "sqlite3"]);
      say("Loading the engine…");
      const files = await (await fetch("engine_bundle.json")).json();
      for (const [path, text] of Object.entries(files)) {
        const full = "/app/" + path;
        pyodide.FS.mkdirTree(full.slice(0, full.lastIndexOf("/")));
        pyodide.FS.writeFile(full, text);
      }
      pyodide.runPython("import sys; sys.path.insert(0, '/app')");
      const dispatch = pyodide.pyimport("engine.bridge").dispatch;
      window.kaeBridge = (method, url, body) =>
        JSON.parse(dispatch(method, url, JSON.stringify(body === undefined ? null : body)));
      document.getElementById("boot")?.remove();
    } catch (err) {
      say("The demo could not start: " + err.message);
      throw err;
    }
  })();
})();
"""

OVERLAY = """
<div id="boot" class="boot" role="status" aria-live="polite">
  <div>
    <strong>Starting the engine in your browser</strong>
    <p id="bootStatus">Loading…</p>
    <p class="muted">This live demo runs the real Python engine on WebAssembly. Nothing is uploaded; data resets when you reload.</p>
  </div>
</div>
"""

OVERLAY_CSS = """
.boot { position: fixed; inset: 0; background: rgba(246,246,244,.92); display: grid; place-items: center; z-index: 20; text-align: center; padding: 24px; }
.boot > div { max-width: 420px; display: grid; gap: 8px; background: #fff; border: 1px solid var(--line); border-radius: var(--radius); padding: 24px; }
.boot #bootStatus { font: 500 13px var(--mono); color: var(--accent); }
"""


def build(out: Path, pyodide_url: str) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    html = (STATIC / "index.html").read_text()
    html = html.replace('href="/static/', 'href="./').replace('src="/static/', 'src="./')
    html = html.replace('<script src="./app.js"></script>', '<script src="./boot.js"></script>\n<script src="./app.js"></script>')
    html = html.replace("</body>", OVERLAY + "</body>")
    if 'src="./boot.js"' not in html:
        raise SystemExit("Could not inject boot.js; did index.html change?")
    (out / "index.html").write_text(html)
    (out / "style.css").write_text((STATIC / "style.css").read_text() + OVERLAY_CSS)
    shutil.copy(STATIC / "app.js", out / "app.js")
    (out / "boot.js").write_text(BOOT_JS % {"pyodide_url": json.dumps(pyodide_url)})
    (out / ".nojekyll").write_text("")

    bundle = {f"engine/{p.name}": p.read_text() for p in sorted((ROOT / "engine").glob("*.py")) if p.name not in SKIP}
    for p in sorted((ROOT / "examples" / "clinic_intake").iterdir()):
        bundle[f"examples/clinic_intake/{p.name}"] = p.read_text()
    (out / "engine_bundle.json").write_text(json.dumps(bundle))
    print(f"Built {out} ({len(bundle)} files bundled, Pyodide from {pyodide_url})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "site")
    parser.add_argument("--pyodide-url", default=PYODIDE_CDN)
    args = parser.parse_args()
    url = args.pyodide_url if args.pyodide_url.endswith("/") else args.pyodide_url + "/"
    build(args.out, url)
