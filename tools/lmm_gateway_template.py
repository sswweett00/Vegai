#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Strict reusable argv templates for private LMM gateway invocations.

A model invocation should never be formed with a shell or by interpolating a
held-out/user request into arbitrary command text.  This module accepts a JSON
string-array command only when the request/evidence paths occupy the direct
values of their corresponding gateway options.  It deliberately owns neither a
model nor a response policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from lmm_contract import FileIdentity, LMMContractError, MAX_JSON_BYTES, file_identity, require, require_unchanged


REQUEST_PLACEHOLDER = "{request}"
EVIDENCE_PLACEHOLDER = "{evidence}"
GROUNDING_MODES = frozenset(("off", "require-citations", "evidence-or-abstain"))


@dataclass(frozen=True, slots=True)
class GatewayCommandTemplate:
    """A hash-bound gateway argv template with exact private-file placeholders."""

    command: tuple[str, ...]
    identity: FileIdentity
    grounding_mode: str


def require_option_placeholder_binding(command: Sequence[str], option: str, placeholder: str, label: str) -> None:
    """Require one literal placeholder to be the sole direct value of an option."""
    positions = [index for index, item in enumerate(command) if item == placeholder]
    require(len(positions) == 1, f"{label} must contain exactly one literal {placeholder!r}")
    position = positions[0]
    require(position > 0 and command[position - 1] == option,
            f"{label} {placeholder!r} must be the direct value of {option}")
    require(not any(item.startswith(f"{option}=") for item in command),
            f"{label} must not use an additional {option}=... option")
    require(command.count(option) == 1, f"{label} must contain exactly one {option}")


def direct_option_value(command: Sequence[str], option: str, label: str, *, required: bool = False) -> str | None:
    """Read one unambiguous ``--option value`` or ``--option=value`` argument."""
    values: list[str] = []
    for index, item in enumerate(command):
        if item == option:
            require(index + 1 < len(command), f"{label} {option} has no value")
            value = command[index + 1]
            require(value and not value.startswith("--"), f"{label} {option} has an invalid value")
            values.append(value)
        elif item.startswith(f"{option}="):
            value = item[len(option) + 1:]
            require(value and not value.startswith("--"), f"{label} {option}= has an invalid value")
            values.append(value)
    require(len(values) <= 1, f"{label} must not repeat {option}")
    if required:
        require(bool(values), f"{label} must include {option}")
    return values[0] if values else None


def load_gateway_command_template(path: Path, *, label: str = "gateway command") -> GatewayCommandTemplate:
    """Load and hash-bind a no-shell gateway command template.

    The contract purposefully requires the direct long-form request/evidence
    flags used by ``lmm_gateway.py``.  This prevents a placeholder from being
    repurposed as a model path, wrapper argument, or a duplicate overridden
    option.  It also prevents this caller from accidentally persisting raw
    request/response text via gateway output flags.
    """
    identity = file_identity(path, label, maximum_bytes=MAX_JSON_BYTES)
    try:
        value = json.loads(path.resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LMMContractError(f"cannot parse {label} {path}: {error}") from error
    require(isinstance(value, list) and value and all(isinstance(item, str) and item for item in value),
            f"{label} must be a nonempty JSON string array")
    command = tuple(value)
    for item in command:
        require(
            (REQUEST_PLACEHOLDER not in item or item == REQUEST_PLACEHOLDER)
            and (EVIDENCE_PLACEHOLDER not in item or item == EVIDENCE_PLACEHOLDER),
            f"{label} placeholders must occupy complete command arguments",
        )
    require_option_placeholder_binding(command, "--request", REQUEST_PLACEHOLDER, label)
    require_option_placeholder_binding(command, "--evidence", EVIDENCE_PLACEHOLDER, label)
    require(command.count("--print-json") == 1,
            f"{label} must include exactly one --print-json so output provenance is machine-readable")
    require(not any(item == "--output" or item.startswith("--output=") for item in command),
            f"{label} must not include --output; the invoking control plane owns report publication")
    require(not any(item == "--include-response-in-report" or item.startswith("--include-response-in-report=") for item in command),
            f"{label} must not include --include-response-in-report; response retention is explicit at the caller")
    grounding_mode = direct_option_value(command, "--grounding-mode", label) or "off"
    require(grounding_mode in GROUNDING_MODES,
            f"{label} has unsupported --grounding-mode {grounding_mode!r}")
    require_unchanged(path, identity, label)
    return GatewayCommandTemplate(command=command, identity=identity, grounding_mode=grounding_mode)


def expand_gateway_command(template: GatewayCommandTemplate, request_path: Path, evidence_path: Path) -> list[str]:
    """Replace only exact private-file placeholders; never invoke a shell."""
    return [
        str(request_path) if item == REQUEST_PLACEHOLDER else str(evidence_path) if item == EVIDENCE_PLACEHOLDER else item
        for item in template.command
    ]


def require_model_argument(template: GatewayCommandTemplate, model_path: Path, *, label: str = "gateway command") -> None:
    """Ensure the command invokes the same immutable manifest selected by policy."""
    raw_model_path = direct_option_value(template.command, "--model", label, required=True)
    require(raw_model_path is not None, f"{label} must include --model")
    try:
        declared = Path(raw_model_path).resolve()
        expected = model_path.resolve()
    except OSError as error:
        raise LMMContractError(f"cannot resolve {label} model path: {error}") from error
    require(declared == expected,
            f"{label} --model path does not match the portfolio candidate model manifest")
