export function firstValue(value) {
    while (Array.isArray(value)) value = value[0];
    return value;
}

export function resolvedOutputValues(payload) {
    const values = Array.isArray(payload) && payload.length === 1 && Array.isArray(payload[0])
        ? payload[0]
        : payload;
    return Array.isArray(values) ? values.map(firstValue) : null;
}

export function preferredLinkedValue(hasLiveValue, liveValue, resolvedValue) {
    if (hasLiveValue) return { hasValue: true, value: liveValue };
    if (resolvedValue !== undefined && resolvedValue !== null) {
        return { hasValue: true, value: resolvedValue };
    }
    return { hasValue: false };
}

export function normalizeWidgetValue(widget, value) {
    value = firstValue(value);
    const widgetType = String(widget.type || "").toLowerCase();

    if (typeof widget.value === "number" || widgetType === "number" || widgetType === "slider") {
        const numberValue = Number(value);
        return Number.isFinite(numberValue) ? { ok: true, value: numberValue } : { ok: false };
    }
    if (typeof widget.value === "boolean" || widgetType === "toggle" || widgetType === "boolean") {
        if (typeof value === "boolean") return { ok: true, value };
        if (typeof value === "number") return { ok: true, value: value !== 0 };
        if (typeof value === "string") {
            const normalized = value.trim().toLowerCase();
            if (normalized === "true" || normalized === "1") return { ok: true, value: true };
            if (normalized === "false" || normalized === "0") return { ok: true, value: false };
        }
        return { ok: false };
    }
    return { ok: true, value: value ?? "" };
}

function updateInputElement(widget) {
    if (!widget.inputEl) return;
    widget.inputEl.value = String(widget.value ?? "");
    if (typeof Event !== "undefined") {
        widget.inputEl.dispatchEvent?.(new Event("input", { bubbles: true }));
        widget.inputEl.dispatchEvent?.(new Event("change", { bubbles: true }));
    }
}

/**
 * Keep a widget's saved fallback distinct from its connected display value.
 * The caller supplies connection state so this remains frontend-version neutral.
 */
export function createWidgetInputSync(widget, { isConnected, onApplied } = {}) {
    if (widget.__mlxInputSync) return widget.__mlxInputSync;
    const connected = () => Boolean(isConnected?.());
    const originalCallback = widget.callback;
    const originalSerializeValue = widget.serializeValue;
    const state = {
        fallback: widget.value,
        applying: false,
        wasConnected: connected(),
    };

    function apply(value) {
        const normalized = normalizeWidgetValue(widget, value);
        if (!normalized.ok || Object.is(widget.value, normalized.value)) return false;
        state.applying = true;
        try {
            widget.value = normalized.value;
            updateInputElement(widget);
            widget.callback?.(widget.value);
        } finally {
            state.applying = false;
        }
        onApplied?.(widget.value);
        return true;
    }

    widget.callback = function (value, ...args) {
        if (!state.applying && !connected()) state.fallback = value;
        return originalCallback?.call(this, value, ...args);
    };

    widget.serializeValue = async function (...args) {
        if (connected()) return state.fallback;
        if (originalSerializeValue) return await originalSerializeValue.apply(this, args);
        return widget.value;
    };

    const controller = {
        captureSavedValue(force = false) {
            if (force || !connected()) state.fallback = widget.value;
        },
        refresh(hasValue = false, value = undefined) {
            const nowConnected = connected();
            if (!nowConnected) {
                const shouldRestore = state.wasConnected;
                state.wasConnected = false;
                return shouldRestore ? apply(state.fallback) : false;
            }
            state.wasConnected = true;
            return hasValue ? apply(value) : false;
        },
        get fallback() {
            return state.fallback;
        },
    };
    widget.__mlxInputSync = controller;
    return controller;
}
