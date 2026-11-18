// Boots the Python engine in the browser with Pyodide.
window.KAE_STATIC = true;
(function () {
  const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.29.5/full/";
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
