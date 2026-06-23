import { app } from "../../scripts/app.js";

// Shared brand styling for the MLX ComfyUI node packs, applied like the shared
// naming snippet: the values live here in the helper, and each pack opts in from
// its own code with `WEB_DIRECTORY = comfyui_mlx_helpers.WEB_DIRECTORY`.
//
// ComfyUI serves this file per pack under /extensions/<pack>/mlx_node_colors.js,
// so we read our own pack folder from the URL and color ONLY that pack's nodes
// (python_module == "custom_nodes.<pack>"). Packs that do not opt in — including
// every third-party pack — are left untouched.
const NODE_COLOR = "#FF5E00";
const NODE_BGCOLOR = "#202026";

const match = import.meta.url.match(/\/extensions\/([^/]+)\//);
const PACK = match ? decodeURIComponent(match[1]) : null;

if (PACK) {
    app.registerExtension({
        name: "comfyui_mlx_helpers.NodeColors." + PACK,
        beforeRegisterNodeDef(nodeType, nodeData) {
            const mod = (nodeData && nodeData.python_module) || "";
            if (mod !== "custom_nodes." + PACK && !mod.endsWith("." + PACK)) return;
            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onCreated ? onCreated.apply(this, arguments) : undefined;
                this.color = NODE_COLOR;
                this.bgcolor = NODE_BGCOLOR;
                return r;
            };
        },
    });
}
