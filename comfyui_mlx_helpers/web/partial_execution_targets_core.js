export const PARTIAL_EXECUTION_TARGETS_INPUT = "_mlx_partial_execution_targets";

export function belongsToExtensionPack(nodeData, pack) {
    const mod = nodeData?.python_module || "";
    return mod === `custom_nodes.${pack}` || mod.endsWith(`.${pack}`);
}

export function acceptsPartialExecutionTargets(nodeData) {
    return Object.hasOwn(
        nodeData?.input?.hidden || {},
        PARTIAL_EXECUTION_TARGETS_INPUT,
    );
}

export function encodePartialExecutionTargets(targets) {
    if (!Array.isArray(targets)) return null;
    return JSON.stringify(targets);
}

export function annotatePartialExecutionTargets(prompt, targets, classTypes) {
    if (!prompt || typeof prompt !== "object") return 0;
    const encoded = encodePartialExecutionTargets(targets);
    let changed = 0;
    for (const node of Object.values(prompt)) {
        if (!node || !classTypes.has(node.class_type)) continue;
        node.inputs ||= {};
        if (encoded === null) {
            if (Object.hasOwn(node.inputs, PARTIAL_EXECUTION_TARGETS_INPUT)) {
                delete node.inputs[PARTIAL_EXECUTION_TARGETS_INPUT];
                changed += 1;
            }
        } else if (node.inputs[PARTIAL_EXECUTION_TARGETS_INPUT] !== encoded) {
            node.inputs[PARTIAL_EXECUTION_TARGETS_INPUT] = encoded;
            changed += 1;
        }
    }
    return changed;
}

export function annotatePartialExecutionTargetsWorkflow(workflow, targets) {
    if (!workflow || typeof workflow !== "object") return 0;
    const encoded = encodePartialExecutionTargets(targets);
    workflow.extra ||= {};
    if (encoded === null) {
        if (Object.hasOwn(workflow.extra, PARTIAL_EXECUTION_TARGETS_INPUT)) {
            delete workflow.extra[PARTIAL_EXECUTION_TARGETS_INPUT];
            return 1;
        }
        return 0;
    }
    if (workflow.extra[PARTIAL_EXECUTION_TARGETS_INPUT] === encoded) return 0;
    workflow.extra[PARTIAL_EXECUTION_TARGETS_INPUT] = encoded;
    return 1;
}

export function installPartialExecutionTargetTransport(api, state) {
    if (state.installed) return state;
    const originalQueuePrompt = api.queuePrompt;
    api.queuePrompt = function (number, prompt, options) {
        annotatePartialExecutionTargets(
            prompt?.output,
            options?.partialExecutionTargets,
            state.classTypes,
        );
        annotatePartialExecutionTargetsWorkflow(
            prompt?.workflow,
            options?.partialExecutionTargets,
        );
        return originalQueuePrompt.apply(this, arguments);
    };
    state.installed = true;
    return state;
}
