#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Serialize authoritative LMM release transitions in a local CAS ledger.

``lmm_release_governor.py`` intentionally produces immutable candidate release
state and traffic-directive artifacts without taking a distributed lock.  That
is correct for an artifact generator, but a deployment controller needs one
canonical head before it can apply a directive.  This stdlib-only companion
stores accepted governor artifacts in a crash-safe, hash-chained local ledger
and applies compare-and-swap (CAS) semantics under an exclusive lock file.

The registry never routes traffic, signs artifacts, contacts a model provider,
or accepts raw prompts, responses, user identifiers, or telemetry.  It is a
single-host serialization boundary, not a distributed consensus or signing
system.  A real multi-node deployment must still anchor the resulting ledger in
a signed/append-only registry and atomically apply the accepted directive in its
own traffic-control trust boundary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence

from lmm_contract import (
    MAX_JSON_BYTES,
    FileIdentity,
    LMMContractError,
    canonical_json_bytes,
    file_identity,
    require,
    require_exact_fields,
    require_unchanged,
    sha256_bytes,
    validate_identifier,
    validate_sha256,
)
from lmm_release_governor import (
    DECISION_SCHEMA,
    GENESIS as RELEASE_GENESIS,
    MAX_CONSUMED_BATCHES,
    MAX_OBSERVATION_BATCH_BYTES,
    ReleasePolicy,
    ReleaseState,
    current_state_outcome,
    load_json_mapping,
    load_policy,
    load_state,
    recheck_policy,
    traffic_directive,
)


REGISTRY_EVENT_SCHEMA = "vegai-lmm-release-registry-event/v1"
REGISTRY_SCHEMA = "vegai-lmm-release-registry/v1"
GENESIS = "GENESIS"
MAX_EVENTS = MAX_CONSUMED_BATCHES + 1  # Initial state plus each governor transition.
MAX_EVENT_LINE_BYTES = 64 * 1024
MAX_LEDGER_BYTES = 16 * 1024 * 1024
MAX_LOCK_TIMEOUT_SECONDS = 300.0
_PHASES = frozenset(("shadow", "canary", "promoted", "rolled_back"))
_OUTCOMES = frozenset((
    "shadow_insufficient_evidence",
    "shadow_pending",
    "shadow_pass",
    "canary_insufficient_evidence",
    "canary_pending",
    "promote",
    "rollback",
))


class ReleaseRegistryError(LMMContractError):
    """An authoritative local release registry contract was violated."""


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    sha256: str
    bytes: int

    def to_mapping(self) -> dict[str, object]:
        return {"sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True, slots=True)
class RegistryEvent:
    sequence: int
    previous_event_sha256: str
    registry_sha256: str
    release_id: str
    policy_sha256: str
    governor_sha256: str
    baseline_identity_sha256: str
    candidate_identity_sha256: str
    state: ArtifactReference
    decision: ArtifactReference
    state_sequence: int
    phase: str
    outcome: str
    traffic: dict[str, object]
    event_sha256: str


def artifact_reference(identity: FileIdentity) -> ArtifactReference:
    return ArtifactReference(sha256=identity.sha256, bytes=identity.bytes)


def parse_bounded_int(value: object, field: str, minimum: int, maximum: int) -> int:
    require(type(value) is int and minimum <= value <= maximum,
            f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def parse_artifact_reference(value: object, field: str) -> ArtifactReference:
    require(isinstance(value, Mapping), f"{field} must be an object")
    require_exact_fields(value, {"sha256", "bytes"}, field)
    return ArtifactReference(
        sha256=validate_sha256(value["sha256"], f"{field}.sha256"),
        bytes=parse_bounded_int(value["bytes"], f"{field}.bytes", 1, MAX_JSON_BYTES),
    )


def parse_string_set(value: object, field: str, allowed: frozenset[str]) -> str:
    require(isinstance(value, str) and value in allowed,
            f"{field} must be one of {', '.join(sorted(allowed))}")
    return value


def parse_traffic_directive(value: object, field: str) -> dict[str, object]:
    require(isinstance(value, Mapping), f"{field} must be an object")
    expected = {
        "mode", "candidate_basis_points", "baseline_basis_points",
        "shadow_candidate_basis_points", "enforcement",
    }
    require_exact_fields(value, expected, field)
    mode = value["mode"]
    allowed_modes = {
        "shadow-mirror-only", "bounded-canary", "candidate-promoted", "baseline-only-rollback",
    }
    require(isinstance(mode, str) and mode in allowed_modes,
            f"{field}.mode is unsupported")
    candidate = parse_bounded_int(value["candidate_basis_points"], f"{field}.candidate_basis_points", 0, 10_000)
    baseline = parse_bounded_int(value["baseline_basis_points"], f"{field}.baseline_basis_points", 0, 10_000)
    shadow = parse_bounded_int(value["shadow_candidate_basis_points"], f"{field}.shadow_candidate_basis_points", 0, 10_000)
    require(candidate + baseline == 10_000, f"{field} user-visible basis points must sum to 10,000")
    require(value["enforcement"] == "external-deployment-platform-required",
            f"{field}.enforcement is unsupported")
    if mode == "shadow-mirror-only":
        require(candidate == 0 and baseline == 10_000 and shadow > 0,
                f"{field} shadow-only directive is inconsistent")
    elif mode == "bounded-canary":
        require(0 < candidate <= 5_000 and baseline == 10_000 - candidate and shadow == 0,
                f"{field} bounded-canary directive is inconsistent")
    elif mode == "candidate-promoted":
        require(candidate == 10_000 and baseline == 0 and shadow == 0,
                f"{field} promoted directive is inconsistent")
    else:
        require(candidate == 0 and baseline == 10_000 and shadow == 0,
                f"{field} rollback directive is inconsistent")
    return {
        "mode": mode,
        "candidate_basis_points": candidate,
        "baseline_basis_points": baseline,
        "shadow_candidate_basis_points": shadow,
        "enforcement": "external-deployment-platform-required",
    }


def event_payload(event: RegistryEvent) -> dict[str, object]:
    return {
        "schema_version": REGISTRY_EVENT_SCHEMA,
        "sequence": event.sequence,
        "previous_event_sha256": event.previous_event_sha256,
        "registry_sha256": event.registry_sha256,
        "release_id": event.release_id,
        "policy_sha256": event.policy_sha256,
        "governor_sha256": event.governor_sha256,
        "baseline_identity_sha256": event.baseline_identity_sha256,
        "candidate_identity_sha256": event.candidate_identity_sha256,
        "state": event.state.to_mapping(),
        "decision": event.decision.to_mapping(),
        "state_sequence": event.state_sequence,
        "phase": event.phase,
        "outcome": event.outcome,
        "traffic_directive": dict(event.traffic),
    }


def event_digest(event: RegistryEvent) -> str:
    return sha256_bytes(canonical_json_bytes(event_payload(event)))


def event_to_mapping(event: RegistryEvent) -> dict[str, object]:
    mapping = event_payload(event)
    mapping["event_sha256"] = event.event_sha256
    return mapping


def event_from_mapping(value: object, number: int, expected_previous: str) -> RegistryEvent:
    field = f"release registry event {number}"
    require(isinstance(value, Mapping), f"{field} must be an object")
    expected = {
        "schema_version", "sequence", "previous_event_sha256", "registry_sha256", "release_id",
        "policy_sha256", "governor_sha256", "baseline_identity_sha256", "candidate_identity_sha256",
        "state", "decision", "state_sequence", "phase", "outcome", "traffic_directive", "event_sha256",
    }
    require_exact_fields(value, expected, field)
    require(value["schema_version"] == REGISTRY_EVENT_SCHEMA, f"{field} has unsupported schema_version")
    sequence = parse_bounded_int(value["sequence"], f"{field}.sequence", 0, MAX_EVENTS - 1)
    require(sequence == number, f"{field}.sequence is not contiguous")
    previous = value["previous_event_sha256"]
    require(previous == expected_previous, f"{field}.previous_event_sha256 breaks the registry hash chain")
    if previous != GENESIS:
        validate_sha256(previous, f"{field}.previous_event_sha256")
    event = RegistryEvent(
        sequence=sequence,
        previous_event_sha256=previous,
        registry_sha256=validate_sha256(value["registry_sha256"], f"{field}.registry_sha256"),
        release_id=validate_identifier(value["release_id"], f"{field}.release_id"),
        policy_sha256=validate_sha256(value["policy_sha256"], f"{field}.policy_sha256"),
        governor_sha256=validate_sha256(value["governor_sha256"], f"{field}.governor_sha256"),
        baseline_identity_sha256=validate_sha256(value["baseline_identity_sha256"], f"{field}.baseline_identity_sha256"),
        candidate_identity_sha256=validate_sha256(value["candidate_identity_sha256"], f"{field}.candidate_identity_sha256"),
        state=parse_artifact_reference(value["state"], f"{field}.state"),
        decision=parse_artifact_reference(value["decision"], f"{field}.decision"),
        state_sequence=parse_bounded_int(value["state_sequence"], f"{field}.state_sequence", 0, 2**63 - 1),
        phase=parse_string_set(value["phase"], f"{field}.phase", _PHASES),
        outcome=parse_string_set(value["outcome"], f"{field}.outcome", _OUTCOMES),
        traffic=parse_traffic_directive(value["traffic_directive"], f"{field}.traffic_directive"),
        event_sha256=validate_sha256(value["event_sha256"], f"{field}.event_sha256"),
    )
    require(event.event_sha256 == event_digest(event), f"{field}.event_sha256 does not bind its payload")
    return event


def reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseRegistryError(f"registry JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def parse_strict_json(payload: bytes, field: str) -> object:
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReleaseRegistryError(f"cannot decode {field} as UTF-8: {error}") from error
    try:
        return json.loads(decoded, object_pairs_hook=reject_duplicate_json_keys)
    except json.JSONDecodeError as error:
        raise ReleaseRegistryError(f"cannot parse {field} as JSON: {error}") from error


def normalize_ledger_path(path: Path) -> Path:
    expanded = path.expanduser()
    require(not expanded.is_symlink(), "release registry ledger path must not be a symlink")
    return expanded.resolve(strict=False)


def iter_events(path: Path) -> Iterator[RegistryEvent]:
    source = normalize_ledger_path(path)
    require(source.is_file(), f"release registry ledger is not a regular file: {source}")
    identity = file_identity(source, "release registry ledger", maximum_bytes=MAX_LEDGER_BYTES)
    expected_previous = GENESIS
    try:
        with source.open("rb") as handle:
            number = 0
            while True:
                raw = handle.readline(MAX_EVENT_LINE_BYTES + 2)
                if not raw:
                    break
                number += 1
                has_lf = raw.endswith(b"\n")
                payload_bytes = len(raw) - 1 if has_lf else len(raw)
                overlong = payload_bytes > MAX_EVENT_LINE_BYTES or (not has_lf and len(raw) == MAX_EVENT_LINE_BYTES + 2)
                if overlong and not has_lf:
                    while raw and not raw.endswith(b"\n"):
                        raw = handle.readline(MAX_EVENT_LINE_BYTES + 2)
                require(not overlong, f"release registry event {number} exceeds {MAX_EVENT_LINE_BYTES} bytes")
                require(has_lf, f"release registry event {number} must terminate with LF")
                raw = raw[:-1]
                require(not raw.endswith(b"\r"), f"release registry event {number} uses CRLF; strict LF is required")
                require(bool(raw.strip()), f"release registry event {number} is blank")
                event = event_from_mapping(parse_strict_json(raw, f"release registry event {number}"), number - 1, expected_previous)
                require(raw == canonical_json_bytes(event_to_mapping(event)),
                        f"release registry event {number} must use canonical JSON")
                expected_previous = event.event_sha256
                yield event
                require(number <= MAX_EVENTS, f"release registry exceeds {MAX_EVENTS} events")
    except OSError as error:
        raise ReleaseRegistryError(f"cannot read release registry ledger {source}: {error}") from error
    require_unchanged(source, identity, "release registry ledger")


def load_events(path: Path) -> tuple[list[RegistryEvent], FileIdentity | None]:
    source = normalize_ledger_path(path)
    if not source.exists():
        return [], None
    events = list(iter_events(source))
    return events, file_identity(source, "release registry ledger", maximum_bytes=MAX_LEDGER_BYTES)


def fsync_directory(directory: Path) -> None:
    flag = getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(str(directory), os.O_RDONLY | flag)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def write_events_atomic(path: Path, events: Sequence[RegistryEvent]) -> FileIdentity:
    require(events and len(events) <= MAX_EVENTS, "release registry must publish between one and maximum events")
    rendered: list[bytes] = []
    total_bytes = 0
    for event in events:
        encoded = canonical_json_bytes(event_to_mapping(event)) + b"\n"
        require(len(encoded) <= MAX_EVENT_LINE_BYTES, "release registry event exceeds line size limit")
        total_bytes += len(encoded)
        require(total_bytes <= MAX_LEDGER_BYTES, "release registry ledger exceeds maximum size")
        rendered.append(encoded)
    resolved = normalize_ledger_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for encoded in rendered:
                handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
        fsync_directory(resolved.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return file_identity(resolved, "published release registry ledger", maximum_bytes=MAX_LEDGER_BYTES)


class RegistryLock:
    """Fail-closed local single-writer guard; stale locks require human review."""

    def __init__(self, ledger: Path, timeout_seconds: float) -> None:
        self.path = ledger.with_name(f".{ledger.name}.lock")
        self.timeout_seconds = timeout_seconds
        self.descriptor: int | None = None
        self.device_inode: tuple[int, int] | None = None

    def __enter__(self) -> "RegistryLock":
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                self.descriptor = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                status = os.fstat(self.descriptor)
                self.device_inode = (status.st_dev, status.st_ino)
                os.write(self.descriptor, f"pid={os.getpid()}\n".encode("ascii"))
                os.fsync(self.descriptor)
                fsync_directory(self.path.parent)
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise ReleaseRegistryError(
                        f"timed out waiting for release registry lock {self.path}; inspect/remove only after owner review"
                    )
                time.sleep(0.05)

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None
        try:
            status = os.lstat(self.path)
        except FileNotFoundError:
            return
        # Do not remove a replacement lock after a local filesystem race; leave
        # it for explicit owner review rather than violating the fail-closed
        # single-writer contract.
        if self.device_inode != (status.st_dev, status.st_ino):
            return
        self.path.unlink()
        fsync_directory(self.path.parent)


def registry_identity() -> FileIdentity:
    return file_identity(Path(__file__), "LMM release registry", maximum_bytes=MAX_JSON_BYTES)


def require_no_symlink(path: Path, field: str) -> Path:
    require(not path.is_symlink(), f"{field} must not be a symlink")
    return path.resolve()


def parse_redacted_identity(value: object, field: str, expected: FileIdentity | None = None,
                            *, maximum_bytes: int = MAX_JSON_BYTES) -> ArtifactReference | None:
    if value is None:
        return None
    require(isinstance(value, Mapping), f"{field} must be null or an object")
    require_exact_fields(value, {"path", "sha256", "bytes"}, field)
    require(value["path"] == "<redacted>", f"{field}.path must be redacted")
    reference = ArtifactReference(
        sha256=validate_sha256(value["sha256"], f"{field}.sha256"),
        bytes=parse_bounded_int(value["bytes"], f"{field}.bytes", 1, maximum_bytes),
    )
    if expected is not None:
        require(reference == artifact_reference(expected), f"{field} does not match its immutable artifact")
    return reference


def load_decision(path: Path) -> tuple[dict[str, Any], FileIdentity]:
    source = require_no_symlink(path, "release decision")
    mapping, identity = load_json_mapping(source, "release decision")
    expected = {
        "schema_version", "release_id", "outcome", "phase", "active_stage", "traffic_directive",
        "dependencies", "policy", "aggregates", "state", "assumptions_and_limits",
    }
    require_exact_fields(mapping, expected, "release decision")
    require(mapping["schema_version"] == DECISION_SCHEMA, "release decision has unsupported schema_version")
    return mapping, identity


def expected_transition(previous: RegistryEvent | None, state: ReleaseState, outcome: str) -> None:
    if previous is None:
        require(state.sequence == 0 and state.previous_state_sha256 == RELEASE_GENESIS,
                "first registry event must register the governor GENESIS state")
        require(outcome in {"shadow_insufficient_evidence", "canary_insufficient_evidence"},
                "first registry event has unsupported initial outcome")
        return
    require(previous.phase not in {"promoted", "rolled_back"},
            "terminal authoritative release state cannot accept another registry transition")
    require(state.sequence == previous.state_sequence + 1,
            "state sequence is not the direct successor of authoritative registry head")
    require(state.previous_state_sha256 == previous.state.sha256,
            "state predecessor does not match authoritative registry head state")
    allowed: dict[str, set[str]] = {
        "shadow": {"shadow", "canary", "rolled_back"},
        "canary": {"canary", "promoted", "rolled_back"},
    }
    require(state.phase in allowed[previous.phase],
            "release phase transition is not allowed from authoritative registry head")
    if outcome == "shadow_pass":
        require(previous.phase == "shadow" and state.phase == "canary",
                "shadow_pass must transition authoritative shadow state to canary")
    elif outcome.startswith("shadow_"):
        require(previous.phase == "shadow" and state.phase == "shadow",
                "shadow pending/insufficient outcome must remain in shadow")
    elif outcome.startswith("canary_"):
        require(previous.phase == "canary" and state.phase == "canary",
                "canary pending/insufficient outcome must remain in canary")
    elif outcome == "promote":
        require(previous.phase == "canary" and state.phase == "promoted",
                "promote must transition authoritative canary state to promoted")
    else:
        require(outcome == "rollback" and state.phase == "rolled_back",
                "terminal rollback outcome must produce rolled_back state")


def validate_decision(policy: ReleasePolicy, state: ReleaseState, state_identity: FileIdentity,
                      decision: Mapping[str, Any], decision_identity: FileIdentity,
                      previous: RegistryEvent | None) -> tuple[str, dict[str, object]]:
    require(validate_identifier(decision["release_id"], "release decision.release_id") == policy.policy_id,
            "release decision.release_id does not match immutable policy")
    outcome = parse_string_set(decision["outcome"], "release decision.outcome", _OUTCOMES)
    require(decision["phase"] == state.phase, "release decision.phase does not match immutable state")
    require(decision["active_stage"] == state.active_stage,
            "release decision.active_stage does not match immutable state")
    # Initial governor state has the internal marker ``initialized``; its paired
    # directive is the current insufficient-evidence outcome. Every later
    # published transition persists the exact emitted outcome as last_decision,
    # including the shadow_pass transition whose new active canary evidence is
    # intentionally still insufficient at the instant canary begins.
    expected_outcome = current_state_outcome(policy, state)[0] if state.sequence == 0 else state.last_decision
    require(outcome == expected_outcome, "release decision.outcome does not match immutable state transition")
    traffic = parse_traffic_directive(decision["traffic_directive"], "release decision.traffic_directive")
    require(traffic == traffic_directive(policy, state, outcome),
            "release decision traffic directive does not match immutable policy/state outcome")

    dependencies = decision["dependencies"]
    require(isinstance(dependencies, Mapping), "release decision.dependencies must be an object")
    expected_dependencies = {
        "governor", "policy", "task", "baseline", "candidate", "prior_state", "new_state", "observation_batch",
    }
    require_exact_fields(dependencies, expected_dependencies, "release decision.dependencies")
    parse_redacted_identity(dependencies["governor"], "release decision.dependencies.governor", policy.governor_identity)
    parse_redacted_identity(dependencies["policy"], "release decision.dependencies.policy", policy.identity)
    parse_redacted_identity(dependencies["task"], "release decision.dependencies.task", policy.task_identity)
    new_state = parse_redacted_identity(dependencies["new_state"], "release decision.dependencies.new_state", state_identity)
    require(new_state is not None, "inspection-only release decision cannot be committed to authoritative registry")
    if previous is None:
        require(dependencies["prior_state"] is None, "initial release decision must not have prior state dependency")
        require(dependencies["observation_batch"] is None,
                "initial release decision must not have an observation batch dependency")
    else:
        prior = parse_redacted_identity(dependencies["prior_state"], "release decision.dependencies.prior_state")
        require(prior is not None and prior == previous.state,
                "release decision prior state does not match authoritative registry head")
        batch = parse_redacted_identity(
            dependencies["observation_batch"],
            "release decision.dependencies.observation_batch",
            maximum_bytes=MAX_OBSERVATION_BATCH_BYTES,
        )
        require(batch is not None, "advanced release decision must bind an observation batch")
        consumed = state.consumed_batches[-1]
        expected_batch = ArtifactReference(sha256=str(consumed["sha256"]), bytes=int(consumed["bytes"]))
        require(batch == expected_batch,
                "release decision observation batch does not match immutable state consumption")
    # Candidate/baseline mappings deliberately contain no paths or raw request data.
    for arm_name, arm in (("baseline", policy.baseline), ("candidate", policy.candidate)):
        rendered = dependencies[arm_name]
        require(isinstance(rendered, Mapping), f"release decision.dependencies.{arm_name} must be an object")
        require(dict(rendered) == arm.identity_mapping(),
                f"release decision.dependencies.{arm_name} does not match immutable policy arm")
    state_summary = decision["state"]
    require(isinstance(state_summary, Mapping), "release decision.state must be an object")
    require_exact_fields(state_summary, {"sequence", "previous_state_sha256", "observations", "consumed_batches"},
                         "release decision.state")
    require(state_summary["sequence"] == state.sequence,
            "release decision.state.sequence does not match immutable state")
    require(state_summary["previous_state_sha256"] == state.previous_state_sha256,
            "release decision.state.previous_state_sha256 does not match immutable state")
    require(isinstance(state_summary["observations"], Mapping), "release decision.state.observations must be an object")
    require(state_summary["consumed_batches"] == len(state.consumed_batches),
            "release decision.state.consumed_batches does not match immutable state")
    require(decision_identity.bytes <= MAX_JSON_BYTES, "release decision exceeds maximum artifact size")
    return outcome, traffic


def build_event(policy: ReleasePolicy, state: ReleaseState, state_identity: FileIdentity,
                decision_identity: FileIdentity, outcome: str, traffic: Mapping[str, object],
                previous: RegistryEvent | None, registry: FileIdentity) -> RegistryEvent:
    event = RegistryEvent(
        sequence=0 if previous is None else previous.sequence + 1,
        previous_event_sha256=GENESIS if previous is None else previous.event_sha256,
        registry_sha256=registry.sha256,
        release_id=policy.policy_id,
        policy_sha256=policy.identity.sha256,
        governor_sha256=policy.governor_identity.sha256,
        baseline_identity_sha256=state.baseline_identity_sha256,
        candidate_identity_sha256=state.candidate_identity_sha256,
        state=artifact_reference(state_identity),
        decision=artifact_reference(decision_identity),
        state_sequence=state.sequence,
        phase=state.phase,
        outcome=outcome,
        traffic=dict(traffic),
        event_sha256="0" * 64,
    )
    return RegistryEvent(
        sequence=event.sequence,
        previous_event_sha256=event.previous_event_sha256,
        registry_sha256=event.registry_sha256,
        release_id=event.release_id,
        policy_sha256=event.policy_sha256,
        governor_sha256=event.governor_sha256,
        baseline_identity_sha256=event.baseline_identity_sha256,
        candidate_identity_sha256=event.candidate_identity_sha256,
        state=event.state,
        decision=event.decision,
        state_sequence=event.state_sequence,
        phase=event.phase,
        outcome=event.outcome,
        traffic=event.traffic,
        event_sha256=event_digest(event),
    )


def expected_event_traffic(policy: ReleasePolicy, phase: str, outcome: str) -> dict[str, object]:
    """Reconstruct the only directive allowed for an accepted ledger event."""
    if outcome == "promote":
        return {
            "mode": "candidate-promoted",
            "candidate_basis_points": 10_000,
            "baseline_basis_points": 0,
            "shadow_candidate_basis_points": 0,
            "enforcement": "external-deployment-platform-required",
        }
    if outcome == "rollback":
        return {
            "mode": "baseline-only-rollback",
            "candidate_basis_points": 0,
            "baseline_basis_points": 10_000,
            "shadow_candidate_basis_points": 0,
            "enforcement": "external-deployment-platform-required",
        }
    if phase == "shadow":
        return {
            "mode": "shadow-mirror-only",
            "candidate_basis_points": 0,
            "baseline_basis_points": 10_000,
            "shadow_candidate_basis_points": policy.shadow_basis_points,
            "enforcement": "external-deployment-platform-required",
        }
    require(phase == "canary", "nonterminal release registry event must be shadow or canary")
    return {
        "mode": "bounded-canary",
        "candidate_basis_points": policy.canary_basis_points,
        "baseline_basis_points": 10_000 - policy.canary_basis_points,
        "shadow_candidate_basis_points": 0,
        "enforcement": "external-deployment-platform-required",
    }


def validate_historical_transition(previous: RegistryEvent | None, event: RegistryEvent,
                                  policy: ReleasePolicy) -> None:
    """Validate cross-event semantics without treating a hash-chain as a signature.

    Historical inspection cannot dereference every state artifact (that would
    make retention/TOCTOU properties of the ledger depend on external paths),
    but it can still reject internally inconsistent phase/outcome/directive
    claims even if a corrupted line has been rehashed.
    """
    field = f"release registry event {event.sequence}"
    if previous is None:
        require(event.state_sequence == 0, "first release registry event must have governor sequence zero")
        if policy.shadow_basis_points > 0:
            require(event.phase == "shadow" and event.outcome == "shadow_insufficient_evidence",
                    "first release registry event is inconsistent with shadow-enabled policy")
        else:
            require(event.phase == "canary" and event.outcome == "canary_insufficient_evidence",
                    "first release registry event is inconsistent with shadow-disabled policy")
    else:
        require(event.state_sequence == previous.state_sequence + 1,
                f"{field} state sequence is not contiguous")
        require(previous.phase not in {"promoted", "rolled_back"},
                f"{field} follows a terminal release registry event")
        allowed: dict[str, set[str]] = {
            "shadow": {"shadow", "canary", "rolled_back"},
            "canary": {"canary", "promoted", "rolled_back"},
        }
        require(event.phase in allowed[previous.phase], f"{field} phase transition is unsupported")
        if event.outcome == "shadow_pass":
            require(previous.phase == "shadow" and event.phase == "canary",
                    f"{field} shadow_pass transition is inconsistent")
        elif event.outcome in {"shadow_insufficient_evidence", "shadow_pending"}:
            require(previous.phase == "shadow" and event.phase == "shadow",
                    f"{field} shadow outcome is inconsistent")
        elif event.outcome in {"canary_insufficient_evidence", "canary_pending"}:
            require(previous.phase == "canary" and event.phase == "canary",
                    f"{field} canary outcome is inconsistent")
        elif event.outcome == "promote":
            require(previous.phase == "canary" and event.phase == "promoted",
                    f"{field} promote transition is inconsistent")
        else:
            require(event.outcome == "rollback" and event.phase == "rolled_back",
                    f"{field} rollback transition is inconsistent")
    require(event.traffic == expected_event_traffic(policy, event.phase, event.outcome),
            f"{field} traffic directive is inconsistent with policy and transition")


def validate_existing_chain(events: Sequence[RegistryEvent], policy: ReleasePolicy, registry: FileIdentity) -> None:
    for index, event in enumerate(events):
        field = f"release registry event {index}"
        require(event.registry_sha256 == registry.sha256,
                f"{field} registry implementation identity does not match supplied registry")
        require(event.release_id == policy.policy_id, f"{field} release_id does not match supplied policy")
        require(event.policy_sha256 == policy.identity.sha256, f"{field} policy identity does not match supplied policy")
        require(event.governor_sha256 == policy.governor_identity.sha256,
                f"{field} governor identity does not match supplied policy")
        require(event.baseline_identity_sha256 == policy.baseline.identity_sha256,
                f"{field} baseline identity does not match supplied policy")
        require(event.candidate_identity_sha256 == policy.candidate.identity_sha256,
                f"{field} candidate identity does not match supplied policy")
        validate_historical_transition(None if index == 0 else events[index - 1], event, policy)


def record_transition(arguments: argparse.Namespace) -> dict[str, object]:
    policy_path = require_no_symlink(arguments.policy, "release policy")
    state_path = require_no_symlink(arguments.state, "release state")
    decision_path = require_no_symlink(arguments.decision, "release decision")
    ledger = normalize_ledger_path(arguments.ledger)
    protected = {str(policy_path), str(state_path), str(decision_path)}
    require(str(ledger) not in protected, "release registry ledger must not overwrite an immutable input")
    registry = registry_identity()
    policy = load_policy(policy_path)
    state, state_identity = load_state(state_path, policy)
    decision, decision_identity = load_decision(decision_path)
    expected_head = arguments.expected_state_sha256
    require(expected_head == GENESIS or validate_sha256(expected_head, "--expected-state-sha256") == expected_head,
            "--expected-state-sha256 must be GENESIS or lowercase SHA-256")
    require(arguments.lock_timeout_seconds >= 0.0 and arguments.lock_timeout_seconds <= MAX_LOCK_TIMEOUT_SECONDS,
            f"--lock-timeout-seconds must be within [0, {MAX_LOCK_TIMEOUT_SECONDS:g}]")

    ledger.parent.mkdir(parents=True, exist_ok=True)
    with RegistryLock(ledger, arguments.lock_timeout_seconds):
        events, ledger_before = load_events(ledger)
        validate_existing_chain(events, policy, registry)
        current_head = GENESIS if not events else events[-1].state.sha256
        require(expected_head == current_head,
                f"release registry head conflict: expected {expected_head}, actual {current_head}; reload authoritative head")
        previous = None if not events else events[-1]
        outcome, traffic = validate_decision(policy, state, state_identity, decision, decision_identity, previous)
        expected_transition(previous, state, outcome)
        recheck_policy(policy)
        require_unchanged(Path(registry.path), registry, "LMM release registry")
        require_unchanged(state_path, state_identity, "release state")
        require_unchanged(decision_path, decision_identity, "release decision")
        if ledger_before is not None:
            require_unchanged(ledger, ledger_before, "release registry ledger")
        event = build_event(policy, state, state_identity, decision_identity, outcome, traffic, previous, registry)
        published = write_events_atomic(ledger, [*events, event])
    return {
        "schema_version": REGISTRY_SCHEMA,
        "release_id": event.release_id,
        "sequence": event.sequence,
        "head_event_sha256": event.event_sha256,
        "head_state_sha256": event.state.sha256,
        "state_sequence": event.state_sequence,
        "phase": event.phase,
        "outcome": event.outcome,
        "traffic_directive": event.traffic,
        "ledger_sha256": published.sha256,
        "ledger_bytes": published.bytes,
    }


def inspect_registry(arguments: argparse.Namespace) -> dict[str, object]:
    policy_path = require_no_symlink(arguments.policy, "release policy")
    ledger = normalize_ledger_path(arguments.ledger)
    policy = load_policy(policy_path)
    events, identity = load_events(ledger)
    require(events and identity is not None, "release registry ledger is empty")
    registry = registry_identity()
    validate_existing_chain(events, policy, registry)
    recheck_policy(policy)
    require_unchanged(Path(registry.path), registry, "LMM release registry")
    require_unchanged(ledger, identity, "release registry ledger")
    head = events[-1]
    return {
        "schema_version": REGISTRY_SCHEMA,
        "events": len(events),
        "head_event_sha256": head.event_sha256,
        "head_state_sha256": head.state.sha256,
        "head_decision_sha256": head.decision.sha256,
        "state_sequence": head.state_sequence,
        "phase": head.phase,
        "outcome": head.outcome,
        "traffic_directive": head.traffic,
        "ledger_sha256": identity.sha256,
        "ledger_bytes": identity.bytes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="CAS-append one verified governor state/decision to authoritative local ledger")
    record.add_argument("--policy", required=True, type=Path)
    record.add_argument("--state", required=True, type=Path)
    record.add_argument("--decision", required=True, type=Path)
    record.add_argument("--ledger", required=True, type=Path)
    record.add_argument("--expected-state-sha256", required=True,
                        help="GENESIS for initial state, otherwise the currently authoritative state SHA-256")
    record.add_argument("--lock-timeout-seconds", type=float, default=5.0)
    inspect = commands.add_parser("inspect", help="verify and print privacy-safe authoritative local registry head")
    inspect.add_argument("--policy", required=True, type=Path)
    inspect.add_argument("--ledger", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "record":
            report = record_transition(arguments)
        elif arguments.command == "inspect":
            report = inspect_registry(arguments)
        else:
            raise ReleaseRegistryError("unsupported release registry command")
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    except (ReleaseRegistryError, LMMContractError, OSError, ValueError) as error:
        print(f"LMM release registry: ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
