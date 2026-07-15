import assert from "node:assert/strict";
import test from "node:test";

import {
    createWidgetInputSync,
    firstValue,
    normalizeWidgetValue,
    preferredLinkedValue,
    resolvedOutputValues,
} from "../comfyui_mlx_helpers/web/widget_input_sync_core.js";


test("firstValue unwraps Comfy UI payload lists", () => {
    assert.equal(firstValue([[["exr"]]]), "exr");
});

test("normalizeWidgetValue retains scalar widget types", () => {
    assert.deepEqual(normalizeWidgetValue({ value: 1, type: "number" }, ["7"]), { ok: true, value: 7 });
    assert.deepEqual(normalizeWidgetValue({ value: false, type: "toggle" }, ["true"]), { ok: true, value: true });
    assert.deepEqual(normalizeWidgetValue({ value: "", type: "text" }, ["hello"]), { ok: true, value: "hello" });
});

test("resolvedOutputValues preserves output slots and unwraps UI values", () => {
    assert.deepEqual(
        resolvedOutputValues([["path.exr", null, [12], [true]]]),
        ["path.exr", null, 12, true],
    );
    assert.equal(resolvedOutputValues({ output: "path.exr" }), null);
});

test("live primitive edits take priority over the last executed output", () => {
    assert.deepEqual(
        preferredLinkedValue(true, "edited", "last execution"),
        { hasValue: true, value: "edited" },
    );
    assert.deepEqual(
        preferredLinkedValue(false, undefined, "computed"),
        { hasValue: true, value: "computed" },
    );
});

test("connected display changes preserve and restore the saved fallback", async () => {
    let connected = false;
    const callbackValues = [];
    const widget = {
        name: "full_path",
        type: "text",
        value: "saved.exr",
        callback(value) {
            callbackValues.push(value);
        },
    };
    const sync = createWidgetInputSync(widget, { isConnected: () => connected });

    widget.value = "edited.exr";
    widget.callback(widget.value);
    assert.equal(sync.fallback, "edited.exr");

    connected = true;
    assert.equal(sync.refresh(true, "upstream.v01.exr"), true);
    assert.equal(widget.value, "upstream.v01.exr");
    assert.equal(await widget.serializeValue(), "edited.exr");

    assert.equal(sync.refresh(true, "upstream.v02.exr"), true);
    assert.equal(widget.value, "upstream.v02.exr");
    assert.equal(sync.fallback, "edited.exr");

    connected = false;
    assert.equal(sync.refresh(), true);
    assert.equal(widget.value, "edited.exr");
    assert.deepEqual(callbackValues, ["edited.exr", "upstream.v01.exr", "upstream.v02.exr", "edited.exr"]);
});
