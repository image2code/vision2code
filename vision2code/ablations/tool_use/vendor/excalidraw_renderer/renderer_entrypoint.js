import { exportToCanvas, restore } from "@excalidraw/excalidraw";

async function renderExcalidrawScene(scene, mount, options = {}) {
  const restored = restore(scene, null, null);
  const appState = {
    ...(restored.appState || {}),
    exportBackground: true,
    viewBackgroundColor:
      (restored.appState && restored.appState.viewBackgroundColor) || "#ffffff",
  };
  const canvas = await exportToCanvas({
    elements: restored.elements || [],
    appState,
    files: restored.files || {},
    getDimensions: () => ({
      width: Number(options.width || appState.width || 1024),
      height: Number(options.height || appState.height || 768),
      scale: 1,
    }),
  });
  mount.innerHTML = "";
  mount.appendChild(canvas);
  return canvas;
}

window.renderExcalidrawScene = renderExcalidrawScene;
