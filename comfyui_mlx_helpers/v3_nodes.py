"""Compatibility adapters for moving stateless ComfyUI V1 nodes to V3.

The model ports historically expose V1 classes through ``NODE_CLASS_MAPPINGS``.
ComfyUI's V3 base class intentionally remains backwards compatible with that
registration path, which lets a mixed node pack migrate its MLX nodes without
also rewriting unrelated bundled V1 nodes.  The adapters produced here:

* retain the serialized node IDs and input/output ordering used by workflows;
* describe the node through :class:`comfy_api.latest.io.Schema`;
* delegate execution to the existing, already-tested implementation; and
* carry resolved scalar inputs and outputs as intermediate UI data without
  turning every compute node into an output node.

The wrapped implementation must be stateless between calls.  Persistent MLX
state belongs on loader outputs/model components (as required by the shared
compile-cache contract), not on a Comfy node instance.
"""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Mapping
from functools import cache
from typing import Any


_WIDGET_INPUT_TYPES = {"STRING", "INT", "FLOAT", "BOOLEAN"}
_HIDDEN_ATTRIBUTE = {
    "UNIQUE_ID": "unique_id",
    "PROMPT": "prompt",
    "EXTRA_PNGINFO": "extra_pnginfo",
    "DYNPROMPT": "dynprompt",
    "AUTH_TOKEN_COMFY_ORG": "auth_token_comfy_org",
    "API_KEY_COMFY_ORG": "api_key_comfy_org",
}


def _io_module():
    try:
        from comfy_api.latest import io
    except ImportError:
        return None
    return io


def v3_nodes_available() -> bool:
    """Return whether the running ComfyUI exposes its V3 node API."""

    return _io_module() is not None


def _type_name(input_type: Any) -> str:
    return input_type if isinstance(input_type, str) else str(input_type)


def _split_input_spec(spec: Any) -> tuple[Any, dict[str, Any]]:
    if not isinstance(spec, (tuple, list)) or not spec:
        raise TypeError(f"Unsupported ComfyUI V1 input specification: {spec!r}")
    options = dict(spec[1]) if len(spec) > 1 and isinstance(spec[1], Mapping) else {}
    return spec[0], options


def _input_from_v1(io, name: str, spec: Any, *, optional: bool):
    input_type, options = _split_input_spec(spec)
    if isinstance(input_type, (tuple, list)):
        # Combo is represented explicitly in V3.  Comfy's V1 compatibility
        # view emits the equivalent COMBO/options form; serialized workflows
        # store widget values by input order, which remains unchanged.
        return io.Combo.Input(
            name,
            options=list(input_type),
            optional=optional,
            extra_dict=options,
        )

    # Custom works for built-ins as well as model-specific sockets.  Keeping
    # the original option mapping in extra_dict avoids silently changing legacy
    # flags such as readonly/disabled/serialize on help widgets.
    return io.Custom(_type_name(input_type)).Input(
        name,
        optional=optional,
        extra_dict=options,
    )


def _schema_hidden(io, input_types: Mapping[str, Any]) -> list[Any]:
    hidden = []
    for hidden_type in (input_types.get("hidden") or {}).values():
        if isinstance(hidden_type, (tuple, list)):
            hidden_type = hidden_type[0] if hidden_type else None
        try:
            hidden.append(io.Hidden(_type_name(hidden_type)))
        except (TypeError, ValueError):
            raise TypeError(f"Unsupported V1 hidden input type: {hidden_type!r}") from None
    return hidden


def _output_from_v1(io, output_type: Any, index: int, legacy_class: type):
    names = getattr(legacy_class, "RETURN_NAMES", ()) or ()
    tooltips = getattr(legacy_class, "OUTPUT_TOOLTIPS", ()) or ()
    list_flags = getattr(legacy_class, "OUTPUT_IS_LIST", ()) or ()
    display_name = names[index] if index < len(names) else _type_name(output_type)
    tooltip = tooltips[index] if index < len(tooltips) else None
    is_output_list = bool(list_flags[index]) if index < len(list_flags) else False
    return io.Custom(_type_name(output_type)).Output(
        id=f"output_{index}",
        display_name=display_name,
        tooltip=tooltip,
        is_output_list=is_output_list,
    )


def _legacy_schema(
    io,
    node_id: str,
    display_name: str,
    legacy_class: type,
    *,
    sync_widget_inputs: bool,
):
    input_types = legacy_class.INPUT_TYPES()
    inputs = []
    for section, optional in (("required", False), ("optional", True)):
        for name, spec in (input_types.get(section) or {}).items():
            inputs.append(_input_from_v1(io, name, spec, optional=optional))

    outputs = [
        _output_from_v1(io, output_type, index, legacy_class)
        for index, output_type in enumerate(getattr(legacy_class, "RETURN_TYPES", ()) or ())
    ]
    return io.Schema(
        node_id=node_id,
        display_name=display_name,
        category=getattr(legacy_class, "CATEGORY", "sd"),
        inputs=inputs,
        outputs=outputs,
        hidden=_schema_hidden(io, input_types),
        description=getattr(legacy_class, "DESCRIPTION", "") or "",
        is_input_list=bool(getattr(legacy_class, "INPUT_IS_LIST", False)),
        is_output_node=bool(getattr(legacy_class, "OUTPUT_NODE", False)),
        is_deprecated=bool(getattr(legacy_class, "DEPRECATED", False)),
        is_experimental=bool(getattr(legacy_class, "EXPERIMENTAL", False)),
        is_dev_only=bool(getattr(legacy_class, "DEV_ONLY", False)),
        not_idempotent=bool(getattr(legacy_class, "NOT_IDEMPOTENT", False)),
        accept_all_inputs=bool(getattr(legacy_class, "ACCEPT_ALL_INPUTS", False)),
        enable_expand=True,
        # Preserve UI result payloads across refresh without making ordinary
        # loader/sampler/intermediate nodes graph roots.
        has_intermediate_output=(
            sync_widget_inputs
            or bool(getattr(legacy_class, "HAS_INTERMEDIATE_OUTPUT", False))
        ),
    )


@cache
def _widget_input_names(legacy_class: type) -> frozenset[str]:
    names = set()
    for section in ("required", "optional"):
        for name, spec in (legacy_class.INPUT_TYPES().get(section) or {}).items():
            if name == "mlx_node_help":
                continue
            input_type, _ = _split_input_spec(spec)
            if isinstance(input_type, (tuple, list)) or _type_name(input_type) in _WIDGET_INPUT_TYPES:
                names.add(name)
    return frozenset(names)


def _json_safe_widget_value(value: Any):
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _hidden_kwargs(adapter_class: type, legacy_class: type) -> dict[str, Any]:
    hidden_specs = legacy_class.INPUT_TYPES().get("hidden") or {}
    values = {}
    for name, hidden_type in hidden_specs.items():
        if isinstance(hidden_type, (tuple, list)):
            hidden_type = hidden_type[0] if hidden_type else None
        attribute = _HIDDEN_ATTRIBUTE.get(_type_name(hidden_type))
        if attribute is not None:
            values[name] = getattr(adapter_class.hidden, attribute, None)
    return values


def _resolved_widget_outputs(legacy_class: type, values: tuple[Any, ...]) -> list[Any]:
    output_types = getattr(legacy_class, "RETURN_TYPES", ()) or ()
    list_flags = getattr(legacy_class, "OUTPUT_IS_LIST", ()) or ()
    resolved = []
    for index, output_type in enumerate(output_types):
        is_list = bool(list_flags[index]) if index < len(list_flags) else False
        if index >= len(values) or is_list or _type_name(output_type) not in _WIDGET_INPUT_TYPES:
            resolved.append(None)
        else:
            resolved.append(_json_safe_widget_value(values[index]))
    return resolved


def _as_node_output(
    io,
    output: Any,
    resolved_inputs: Mapping[str, Any],
    legacy_class: type,
):
    if isinstance(output, io.NodeOutput):
        node_output = output
    elif output is None:
        node_output = io.NodeOutput()
    elif isinstance(output, tuple):
        node_output = io.NodeOutput(*output)
    elif isinstance(output, dict):
        node_output = io.NodeOutput.from_dict(output)
    else:
        # Let ComfyUI's normalizer report specialized values such as an
        # ExecutionBlocker correctly instead of obscuring them here.
        return output

    if not resolved_inputs:
        return node_output
    ui_payload = dict(node_output.ui or {})
    ui_payload["mlx_resolved_inputs"] = [dict(resolved_inputs)]
    ui_payload["mlx_resolved_outputs"] = [
        _resolved_widget_outputs(legacy_class, node_output.args)
    ]
    return io.NodeOutput(
        *node_output.args,
        ui=ui_payload,
        expand=node_output.expand,
        block_execution=node_output.block_execution,
    )


def _execute_legacy(
    adapter_class: type,
    legacy_class: type,
    kwargs: dict[str, Any],
    *,
    sync_widget_inputs: bool,
):
    values = dict(kwargs)
    values.update(_hidden_kwargs(adapter_class, legacy_class))
    implementation = legacy_class()
    function = getattr(implementation, legacy_class.FUNCTION)
    output = function(**values)
    resolved = {}
    if sync_widget_inputs:
        resolved = {
            name: [_json_safe_widget_value(kwargs[name])]
            for name in _widget_input_names(legacy_class)
            if name in kwargs
        }
    return output, resolved


def adapt_v1_node(
    node_id: str,
    legacy_class: type,
    *,
    display_name: str | None = None,
    sync_widget_inputs: bool = False,
) -> type:
    """Create a V3 schema-backed class delegating to a stateless V1 node."""

    io = _io_module()
    if io is None:
        return legacy_class

    display_name = display_name or node_id

    def define_schema(cls):
        return _legacy_schema(
            io,
            node_id,
            display_name,
            legacy_class,
            sync_widget_inputs=sync_widget_inputs,
        )

    legacy_function = getattr(legacy_class, legacy_class.FUNCTION)
    if inspect.iscoroutinefunction(legacy_function):
        async def execute(cls, **kwargs):
            values = dict(kwargs)
            values.update(_hidden_kwargs(cls, legacy_class))
            output = await getattr(legacy_class(), legacy_class.FUNCTION)(**values)
            resolved = {}
            if sync_widget_inputs:
                resolved = {
                    name: [_json_safe_widget_value(kwargs[name])]
                    for name in _widget_input_names(legacy_class)
                    if name in kwargs
                }
            return _as_node_output(io, output, resolved, legacy_class)
    else:
        def execute(cls, **kwargs):
            output, resolved = _execute_legacy(
                cls,
                legacy_class,
                kwargs,
                sync_widget_inputs=sync_widget_inputs,
            )
            return _as_node_output(io, output, resolved, legacy_class)

    class_name = re.sub(r"\W+", "_", f"{node_id}V3")
    attributes = {
        "__module__": legacy_class.__module__,
        "__doc__": legacy_class.__doc__,
        "LEGACY_NODE_CLASS": legacy_class,
        "V3_ADAPTER": True,
        "define_schema": classmethod(define_schema),
        "execute": classmethod(execute),
    }

    for old_name, new_name in (
        ("VALIDATE_INPUTS", "validate_inputs"),
        ("IS_CHANGED", "fingerprint_inputs"),
    ):
        hook = getattr(legacy_class, old_name, None)
        if hook is not None:
            def forwarded(cls, _hook=hook, **kwargs):
                return _hook(**kwargs)
            attributes[new_name] = classmethod(forwarded)

    if "check_lazy_status" in legacy_class.__dict__:
        def check_lazy_status(cls, **kwargs):
            return legacy_class().check_lazy_status(**kwargs)
        attributes["check_lazy_status"] = classmethod(check_lazy_status)

    return type(class_name, (io.ComfyNode,), attributes)


def adapt_v1_nodes(
    node_class_mappings: Mapping[str, type],
    display_name_mappings: Mapping[str, str] | None = None,
    *,
    sync_widget_inputs: bool = False,
) -> dict[str, type]:
    """Return a mapping with every V1 class adapted when V3 is available.

    On older ComfyUI builds the original mapping is returned, preserving the
    pre-migration runtime path.
    """

    if not v3_nodes_available():
        return dict(node_class_mappings)
    display_name_mappings = display_name_mappings or {}
    return {
        node_id: adapt_v1_node(
            node_id,
            legacy_class,
            display_name=display_name_mappings.get(node_id, node_id),
            sync_widget_inputs=sync_widget_inputs,
        )
        for node_id, legacy_class in node_class_mappings.items()
    }


__all__ = ["adapt_v1_node", "adapt_v1_nodes", "v3_nodes_available"]
