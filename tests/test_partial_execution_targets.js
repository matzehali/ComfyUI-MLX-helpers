import assert from "node:assert/strict";
import test from "node:test";

import {
    PARTIAL_EXECUTION_TARGETS_INPUT,
    acceptsPartialExecutionTargets,
    annotatePartialExecutionTargets,
    annotatePartialExecutionTargetsWorkflow,
    belongsToExtensionPack,
    installPartialExecutionTargetTransport,
} from "../comfyui_mlx_helpers/web/partial_execution_targets_core.js";

test("node definitions opt into partial target transport with a hidden input", () => {
    assert.equal(acceptsPartialExecutionTargets({
        input: { hidden: { [PARTIAL_EXECUTION_TARGETS_INPUT]: "STRING" } },
    }), true);
    assert.equal(acceptsPartialExecutionTargets({ input: { hidden: {} } }), false);
});

test("node registration is scoped to the extension pack serving the transport", () => {
    assert.equal(
        belongsToExtensionPack(
            { python_module: "custom_nodes.comfyui-sidecarjson" },
            "comfyui-sidecarjson",
        ),
        true,
    );
    assert.equal(
        belongsToExtensionPack({ python_module: "custom_nodes.other" }, "comfyui-sidecarjson"),
        false,
    );
});

test("only opted-in prompt nodes receive serialized partial roots", () => {
    const prompt = {
        sidecar: { class_type: "Sidecar", inputs: { full_path: "render.exr" } },
        other: { class_type: "Other", inputs: {} },
    };
    assert.equal(annotatePartialExecutionTargets(prompt, ["preview"], new Set(["Sidecar"])), 1);
    assert.equal(prompt.sidecar.inputs[PARTIAL_EXECUTION_TARGETS_INPUT], '["preview"]');
    assert.equal(Object.hasOwn(prompt.other.inputs, PARTIAL_EXECUTION_TARGETS_INPUT), false);
});

test("queue transport forwards targets and preserves the original API call", async () => {
    const calls = [];
    const api = {
        async queuePrompt(...args) {
            calls.push({ receiver: this, args });
            return "queued";
        },
    };
    const state = { classTypes: new Set(["Sidecar"]), installed: false };
    installPartialExecutionTargetTransport(api, state);
    installPartialExecutionTargetTransport(api, state);
    const request = {
        output: { sidecar: { class_type: "Sidecar", inputs: {} } },
        workflow: {},
    };

    assert.equal(await api.queuePrompt(0, request, { partialExecutionTargets: ["preview"] }), "queued");
    assert.equal(calls.length, 1);
    assert.equal(calls[0].receiver, api);
    assert.equal(calls[0].args[0], 0);
    assert.equal(
        request.output.sidecar.inputs[PARTIAL_EXECUTION_TARGETS_INPUT],
        '["preview"]',
    );
    assert.equal(
        request.workflow.extra[PARTIAL_EXECUTION_TARGETS_INPUT],
        '["preview"]',
    );
});

test("workflow metadata transports roots for V3 hidden inputs", () => {
    const workflow = {};
    assert.equal(annotatePartialExecutionTargetsWorkflow(workflow, ["preview"]), 1);
    assert.equal(workflow.extra[PARTIAL_EXECUTION_TARGETS_INPUT], '["preview"]');
    assert.equal(annotatePartialExecutionTargetsWorkflow(workflow, undefined), 1);
    assert.equal(Object.hasOwn(workflow.extra, PARTIAL_EXECUTION_TARGETS_INPUT), false);
});

test("full execution removes a stale partial-target annotation", () => {
    const prompt = {
        sidecar: {
            class_type: "Sidecar",
            inputs: { [PARTIAL_EXECUTION_TARGETS_INPUT]: '["old-preview"]' },
        },
    };
    annotatePartialExecutionTargets(prompt, undefined, new Set(["Sidecar"]));
    assert.equal(Object.hasOwn(prompt.sidecar.inputs, PARTIAL_EXECUTION_TARGETS_INPUT), false);
});
