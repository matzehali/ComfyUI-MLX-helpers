"""Declarative, output-aware lazy-input tracing for ComfyUI nodes.

ComfyUI's scheduler normally follows every linked non-lazy input of a node,
even when the queued downstream path consumes only one cheap metadata output.
This module lets a node declare which inputs each output actually depends on,
then traces the submitted graph backwards from output nodes to request only the
lazy inputs used by those output paths.

Unknown nodes deliberately remain conservative: all of their linked inputs are
traversed.  The helper never guesses dependencies for third-party code.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from typing import Any


OutputDependencies = Mapping[int, Sequence[str]]
PARTIAL_EXECUTION_TARGETS_INPUT = "_mlx_partial_execution_targets"


def parse_partial_execution_targets(value: Any) -> tuple[Any, ...] | None:
    """Decode the frontend-provided partial execution roots.

    ``None`` means that no partial-execution information reached the backend,
    so callers must keep the conservative all-output-roots fallback.  Invalid
    payloads fail the same safe way rather than pruning execution accidentally.
    """

    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return None
    if not isinstance(value, (list, tuple)):
        return None
    if any(
        not isinstance(item, (str, int)) or isinstance(item, bool)
        for item in value
    ):
        return None
    return tuple(value)


def _prompt_node(prompt: Mapping[Any, Any], node_id: Any) -> tuple[Any, Mapping[str, Any]] | None:
    if node_id in prompt:
        return node_id, prompt[node_id]
    text_id = str(node_id)
    if text_id in prompt:
        return text_id, prompt[text_id]
    return None


def _link(value: Any, prompt: Mapping[Any, Any]) -> tuple[Any, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    upstream = _prompt_node(prompt, value[0])
    if upstream is None or not isinstance(value[1], int) or isinstance(value[1], bool):
        return None
    return upstream[0], value[1]


def _class_mappings(class_mappings: Mapping[str, type] | None) -> Mapping[str, type]:
    if class_mappings is not None:
        return class_mappings
    try:
        import nodes
    except ImportError:
        return {}
    return getattr(nodes, "NODE_CLASS_MAPPINGS", {})


def _input_names(class_def: type) -> tuple[str, ...]:
    input_types = class_def.INPUT_TYPES()
    return tuple(
        name
        for section in ("required", "optional")
        for name in (input_types.get(section) or {})
    )


def validate_output_dependencies(
    class_def: type,
    dependencies: OutputDependencies | None = None,
) -> dict[int, tuple[str, ...]]:
    """Validate and normalize a complete output-to-input dependency map."""

    if dependencies is None:
        dependencies = getattr(class_def, "OUTPUT_INPUT_DEPENDENCIES", None)
    if dependencies is None:
        raise ValueError(f"{class_def.__name__} has no OUTPUT_INPUT_DEPENDENCIES declaration")

    output_count = len(getattr(class_def, "RETURN_TYPES", ()) or ())
    expected = set(range(output_count))
    actual = set(dependencies)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{class_def.__name__} output dependency keys must be {sorted(expected)}; "
            f"missing={missing}, extra={extra}"
        )

    known_inputs = set(_input_names(class_def))
    normalized: dict[int, tuple[str, ...]] = {}
    for output_index, names in dependencies.items():
        if isinstance(names, str):
            raise TypeError(
                f"{class_def.__name__} output {output_index} dependencies must be a sequence, not a string"
            )
        names = tuple(dict.fromkeys(names))
        unknown = sorted(set(names) - known_inputs)
        if unknown:
            raise ValueError(
                f"{class_def.__name__} output {output_index} references unknown inputs: {unknown}"
            )
        normalized[output_index] = names
    return normalized


def mark_traced_inputs_lazy(
    input_types: Mapping[str, Any],
    dependencies: OutputDependencies,
) -> dict[str, Any]:
    """Return an INPUT_TYPES copy with every declared data dependency lazy."""

    result = deepcopy(dict(input_types))
    traced_names = {name for names in dependencies.values() for name in names}
    for section in ("required", "optional"):
        specs = result.get(section) or {}
        for name in traced_names.intersection(specs):
            spec = specs[name]
            if not isinstance(spec, (tuple, list)) or not spec:
                raise TypeError(f"Unsupported ComfyUI input specification for {name}: {spec!r}")
            options = dict(spec[1]) if len(spec) > 1 and isinstance(spec[1], Mapping) else {}
            options["lazy"] = True
            specs[name] = (spec[0], options)
    return result


def _dependencies_for(
    class_def: type | None,
    requested_outputs: set[int] | None,
    node_inputs: Mapping[str, Any],
) -> tuple[str, ...]:
    if class_def is None or requested_outputs is None:
        return tuple(node_inputs)

    declaration = getattr(class_def, "OUTPUT_INPUT_DEPENDENCIES", None)
    if declaration is None or any(index not in declaration for index in requested_outputs):
        return tuple(node_inputs)

    declaration = validate_output_dependencies(class_def, declaration)

    return tuple(
        dict.fromkeys(
            name
            for index in sorted(requested_outputs)
            for name in declaration[index]
            if name in node_inputs
        )
    )


def trace_requested_outputs(
    prompt: Mapping[Any, Mapping[str, Any]],
    *,
    class_mappings: Mapping[str, type] | None = None,
    output_node_ids: Sequence[Any] | None = None,
) -> dict[Any, frozenset[int]]:
    """Trace output sockets required by the prompt's executable output paths.

    ``output_node_ids`` narrows the roots when the shared frontend transport has
    forwarded ComfyUI's partial-execution targets.  The safe fallback remains
    every registered ``OUTPUT_NODE`` in the prompt.
    """

    mappings = _class_mappings(class_mappings)
    if output_node_ids is None:
        roots = []
        for node_id, node in prompt.items():
            class_def = mappings.get(node.get("class_type"))
            if class_def is not None and bool(getattr(class_def, "OUTPUT_NODE", False)):
                roots.append(node_id)
    else:
        roots = [resolved[0] for node_id in output_node_ids if (resolved := _prompt_node(prompt, node_id))]

    requested: dict[Any, set[int]] = defaultdict(set)
    processed: dict[Any, set[int]] = defaultdict(set)
    processed_roots: set[Any] = set()
    queue = deque((node_id, None) for node_id in roots)

    while queue:
        node_id, output_indexes = queue.popleft()
        resolved = _prompt_node(prompt, node_id)
        if resolved is None:
            continue
        node_id, node = resolved

        if output_indexes is None:
            if node_id in processed_roots:
                continue
            processed_roots.add(node_id)
        else:
            new_indexes = set(output_indexes) - processed[node_id]
            if not new_indexes:
                continue
            requested[node_id].update(new_indexes)
            processed[node_id].update(new_indexes)

        node_inputs = node.get("inputs") or {}
        class_def = mappings.get(node.get("class_type"))
        dependency_names = _dependencies_for(class_def, output_indexes, node_inputs)
        for input_name in dependency_names:
            link = _link(node_inputs.get(input_name), prompt)
            if link is not None:
                queue.append((link[0], {link[1]}))

    return {node_id: frozenset(indexes) for node_id, indexes in requested.items()}


def requested_outputs_for_node(
    prompt: Mapping[Any, Mapping[str, Any]] | None,
    unique_id: Any,
    *,
    class_mappings: Mapping[str, type] | None = None,
    output_node_ids: Sequence[Any] | None = None,
) -> frozenset[int]:
    """Return the current node's output sockets used by executable paths."""

    if not prompt:
        return frozenset()
    resolved = _prompt_node(prompt, unique_id)
    if resolved is None:
        return frozenset()
    return trace_requested_outputs(
        prompt,
        class_mappings=class_mappings,
        output_node_ids=output_node_ids,
    ).get(resolved[0], frozenset())


def required_inputs_for_node(
    prompt: Mapping[Any, Mapping[str, Any]] | None,
    unique_id: Any,
    class_def: type,
    *,
    class_mappings: Mapping[str, type] | None = None,
    output_node_ids: Sequence[Any] | None = None,
) -> list[str]:
    """Return the declared lazy inputs needed for this node's used outputs."""

    dependencies = validate_output_dependencies(class_def)
    requested = requested_outputs_for_node(
        prompt,
        unique_id,
        class_mappings=class_mappings,
        output_node_ids=output_node_ids,
    )
    if not requested:
        # A scheduled node omitted from the trace must remain correct.  This can
        # happen when a frontend/core adds a new execution-root mechanism.
        return list(dict.fromkeys(name for names in dependencies.values() for name in names))
    return list(
        dict.fromkeys(name for index in sorted(requested) for name in dependencies[index])
    )
