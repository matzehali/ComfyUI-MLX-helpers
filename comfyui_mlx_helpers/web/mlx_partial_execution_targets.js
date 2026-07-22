import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
    acceptsPartialExecutionTargets,
    belongsToExtensionPack,
    installPartialExecutionTargetTransport,
} from "./partial_execution_targets_core.js";

const match = import.meta.url.match(/\/extensions\/([^/]+)\//);
const PACK = match ? decodeURIComponent(match[1]) : null;
const TARGET_STATE = Symbol.for("comfyui_mlx_helpers.partialExecutionTargets");
const targetState = globalThis[TARGET_STATE] ||= {
    classTypes: new Set(),
    installed: false,
};

installPartialExecutionTargetTransport(api, targetState);

if (PACK) {
    app.registerExtension({
        name: `comfyui_mlx_helpers.PartialExecutionTargets.${PACK}`,
        beforeRegisterNodeDef(_nodeType, nodeData) {
            if (
                belongsToExtensionPack(nodeData, PACK)
                && acceptsPartialExecutionTargets(nodeData)
            ) {
                targetState.classTypes.add(nodeData.name);
            }
        },
    });
}
