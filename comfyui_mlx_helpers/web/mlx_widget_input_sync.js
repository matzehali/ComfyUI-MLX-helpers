import { app } from "../../scripts/app.js";
import {
    createWidgetInputSync,
    firstValue,
    preferredLinkedValue,
    resolvedOutputValues,
} from "./widget_input_sync_core.js";

const match = import.meta.url.match(/\/extensions\/([^/]+)\//);
const PACK = match ? decodeURIComponent(match[1]) : null;
const STATE = Symbol("mlxWidgetInputSync");
const WIDGET_TYPES = new Set(["STRING", "INT", "FLOAT", "BOOLEAN", "COMBO"]);

function belongsToPack(nodeData) {
    const mod = nodeData?.python_module || "";
    return mod === `custom_nodes.${PACK}` || mod.endsWith(`.${PACK}`);
}

function widgetInputNames(nodeData) {
    const names = [];
    for (const section of ["required", "optional"]) {
        for (const [name, spec] of Object.entries(nodeData?.input?.[section] || {})) {
            const rawType = Array.isArray(spec) ? spec[0] : spec;
            const type = Array.isArray(rawType) ? "COMBO" : String(rawType || "");
            if (WIDGET_TYPES.has(type) && name !== "mlx_node_help") names.push(name);
        }
    }
    return names;
}

function inputSlot(node, name) {
    return node.inputs?.find((input) => input.name === name);
}

function linkedSource(node, name) {
    const input = inputSlot(node, name);
    if (!input || input.link == null) return null;
    let link = app.graph?.links?.[input.link];
    let guard = 0;
    while (link && guard++ < 32) {
        const source = app.graph.getNodeById(link.origin_id);
        if (!source) return null;
        if ((source.type || source.comfyClass) !== "Reroute") return { source, link };
        const rerouteInput = source.inputs?.[0];
        link = rerouteInput?.link == null ? null : app.graph.links[rerouteInput.link];
    }
    return null;
}

function liveLinkedValue(node, name) {
    const linked = linkedSource(node, name);
    if (!linked) return { hasValue: false };
    const { source, link } = linked;
    const outputName = source.outputs?.[link.origin_slot]?.name;
    const widgets = source.widgets || [];
    const byOutput = widgets.find(
        (widget) => String(widget.name || "").toLowerCase() === String(outputName || "").toLowerCase(),
    );
    const conventional = widgets.find((widget) =>
        ["value", "text", "string", "path"].includes(String(widget.name || "").toLowerCase()),
    );
    const scalarWidgets = widgets.filter((widget) =>
        ["string", "number", "boolean"].includes(typeof widget.value),
    );
    const singleOutputWidget = source.outputs?.length === 1 && scalarWidgets.length === 1
        ? scalarWidgets[0]
        : null;
    const widget = byOutput || conventional || singleOutputWidget;
    const resolvedOutputs = source.__mlxResolvedOutputs;
    const resolvedValue = Array.isArray(resolvedOutputs)
        ? resolvedOutputs[link.origin_slot]
        : undefined;
    return preferredLinkedValue(Boolean(widget), widget?.value, resolvedValue);
}

function refreshNode(node, resolvedInputs = null) {
    const state = node[STATE];
    if (!state) return;
    let changed = false;
    for (const [name, controller] of state.controllers) {
        if (resolvedInputs && Object.hasOwn(resolvedInputs, name)) {
            changed = controller.refresh(true, firstValue(resolvedInputs[name])) || changed;
            continue;
        }
        const live = liveLinkedValue(node, name);
        changed = controller.refresh(live.hasValue, live.value) || changed;
    }
    if (changed) {
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
    }
}

function setupNode(node, names) {
    if (node[STATE]) return;
    const controllers = new Map();
    for (const name of names) {
        const widget = node.widgets?.find((item) => item.name === name);
        if (!widget) continue;
        const controller = createWidgetInputSync(widget, {
            isConnected: () => inputSlot(node, name)?.link != null,
            onApplied: () => node.setDirtyCanvas?.(true, true),
        });
        controllers.set(name, controller);
    }
    node[STATE] = { controllers, lastDrawRefresh: 0 };
    setTimeout(() => refreshNode(node), 0);
}

if (PACK) {
    app.registerExtension({
        name: `comfyui_mlx_helpers.WidgetInputSync.${PACK}`,
        beforeRegisterNodeDef(nodeType, nodeData) {
            if (!belongsToPack(nodeData)) return;
            const names = widgetInputNames(nodeData);
            if (!names.length) return;

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                setupNode(this, names);
                return result;
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const result = onConfigure?.apply(this, arguments);
                setupNode(this, names);
                for (const controller of this[STATE].controllers.values()) {
                    controller.captureSavedValue(true);
                }
                setTimeout(() => refreshNode(this), 0);
                return result;
            };

            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function () {
                const result = onConnectionsChange?.apply(this, arguments);
                setTimeout(() => refreshNode(this), 0);
                return result;
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                const result = onExecuted?.apply(this, arguments);
                const resolvedOutputs = resolvedOutputValues(message?.mlx_resolved_outputs);
                if (resolvedOutputs) this.__mlxResolvedOutputs = resolvedOutputs;
                const resolved = message?.mlx_resolved_inputs?.[0];
                refreshNode(this, resolved && typeof resolved === "object" ? resolved : null);
                return result;
            };

            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function () {
                const now = performance.now();
                if (!this[STATE] || now - this[STATE].lastDrawRefresh > 50) {
                    setupNode(this, names);
                    this[STATE].lastDrawRefresh = now;
                    refreshNode(this);
                }
                return onDrawForeground?.apply(this, arguments);
            };
        },
    });
}
