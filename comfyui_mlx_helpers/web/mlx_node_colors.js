import { app } from "../../scripts/app.js";

// Shared brand styling for the MLX ComfyUI node packs.
// Loaded once (any pack that points WEB_DIRECTORY at comfyui_mlx_helpers/web picks
// it up), and applied to every custom node — python_module under "custom_nodes",
// i.e. not the built-in core nodes.
const NODE_COLOR = "#FF5E00";
const NODE_BGCOLOR = "#202026";

if (!window.__mlxNodeColors) {
    window.__mlxNodeColors = true;
    app.registerExtension({
        name: "comfyui_mlx_helpers.NodeColors",
        beforeRegisterNodeDef(nodeType, nodeData) {
            const mod = (nodeData && nodeData.python_module) || "";
            if (!mod.startsWith("custom_nodes")) return;
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
