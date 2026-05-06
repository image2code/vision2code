# Excalidraw Renderer Bundle

`render_excalidraw.py` expects a local browser bundle at:

```text
tool_use_ablation/vendor/excalidraw_renderer/renderer_bundle.js
```

Build it once in an environment with a modern Node/npm stack:

```bash
cd vision2code/ablations/tool_use/vendor/excalidraw_renderer
npm install
npm run build
```

Runtime rendering uses only the checked-out `renderer_bundle.js` plus a Chrome or Chromium binary.
Rendering intentionally fails if the bundle or Chrome/Chromium is missing.
