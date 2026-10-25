#!/usr/bin/env python3
"""Independent oracle for Vegai's tokenizer-free byte-event signal field.

This verifier is intentionally dependency-free and does not import production
Fortran code.  It models raw octet event construction, exact finite
Walsh-Hadamard identities, and (when requested) the compiled CLI contract.

It verifies only bounded digital representation invariants.  It does *not*
validate language meaning, multimedia understanding, external observations,
or physical-world truth.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

BITS = 16_384
WORDS = 256
WORD_BITS = 64
MASK64 = (1 << 64) - 1
MAX_BYTES = 32_767
EXPECTED_ENERGY = BITS * BITS

# These are fixed representation domain separators, not a vocabulary, corpus,
# prompt set, learned parameter set, or hidden training data.
BYTE_TAG = 0xBA5EBA11C0DEC0DE
ORDER_TAG = 0xF13D0FD300000001
TIE_TAG = 0x71E5EED5A17F1E1D
GAMMA = 0x9E3779B97F4A7C15
MUL1 = 0xBF58476D1CE4E5B9
MUL2 = 0x94D049BB133111EB


Vector = tuple[int, ...]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def rotl64(value: int, amount: int) -> int:
    """Rotate a 64-bit bit pattern, including negative rotation counts."""
    amount %= WORD_BITS
    value &= MASK64
    return ((value << amount) | (value >> ((WORD_BITS - amount) % WORD_BITS))) & MASK64


def splitmix_next(state: int) -> tuple[int, int]:
    state = (state + GAMMA) & MASK64
    mixed = state
    mixed = ((mixed ^ (mixed >> 30)) * MUL1) & MASK64
    mixed = ((mixed ^ (mixed >> 27)) * MUL2) & MASK64
    return state, (mixed ^ (mixed >> 31)) & MASK64


def vector_from_seed(seed: int) -> Vector:
    state = seed & MASK64
    words: list[int] = []
    for _ in range(WORDS):
        state, word = splitmix_next(state)
        words.append(word)
    return tuple(words)


def xor_bind(left: Vector, right: Vector) -> Vector:
    return tuple(a ^ b for a, b in zip(left, right, strict=True))


def byte_atom(octet: int) -> Vector:
    require(0 <= octet <= 255, "octet out of range")
    seed = BYTE_TAG ^ rotl64(octet, 17) ^ rotl64(octet, -9)
    return vector_from_seed(seed)


def ordered_byte_atom(octet: int, position_one_based: int) -> Vector:
    require(0 <= octet <= 255 and position_one_based > 0, "invalid byte event")
    seed = ORDER_TAG ^ rotl64(position_one_based, 23)
    seed ^= rotl64(octet, -17)
    seed ^= rotl64(position_one_based ^ octet, 41)
    return vector_from_seed(seed)


def majority_bundle(vectors: list[Vector], tie_seed: int) -> Vector:
    """Direct bipolar-majority oracle, structurally separate from Fortran's accumulator."""
    require(vectors, "cannot bundle an empty event sequence")
    tie = vector_from_seed(tie_seed)
    result = [0] * WORDS
    for word_index in range(WORDS):
        output_word = 0
        for bit in range(WORD_BITS):
            signed_count = sum(
                1 if (vector[word_index] >> bit) & 1 else -1
                for vector in vectors
            )
            if signed_count > 0 or (signed_count == 0 and ((tie[word_index] >> bit) & 1)):
                output_word |= 1 << bit
        result[word_index] = output_word
    return tuple(result)


def encode_raw_bytes(raw: bytes) -> tuple[Vector, Vector]:
    """Return (order_sensitive_field, order_insensitive_distribution_field)."""
    require(0 < len(raw) <= MAX_BYTES, "raw byte sequence exceeds checked field bounds")
    order = tuple(0 for _ in range(WORDS))
    distribution_events: list[Vector] = []
    for position, octet in enumerate(raw, start=1):
        distribution_events.append(byte_atom(octet))
        order = xor_bind(order, ordered_byte_atom(octet, position))
    return order, majority_bundle(distribution_events, TIE_TAG)


def unpack_bipolar(field: Vector) -> list[int]:
    values: list[int] = []
    for word in field:
        values.extend(1 if (word >> bit) & 1 else -1 for bit in range(WORD_BITS))
    require(len(values) == BITS, "unexpected packed field dimension")
    return values


def walsh_forward(field: Vector) -> list[int]:
    return walsh_in_place(unpack_bipolar(field))


def walsh_in_place(values: list[int]) -> list[int]:
    """Unnormalised fast Walsh-Hadamard transform over exact Python integers."""
    require(len(values) == BITS, "unexpected Walsh dimension")
    work = list(values)
    span = 1
    while span < len(work):
        for start in range(0, len(work), 2 * span):
            for offset in range(span):
                left = work[start + offset]
                right = work[start + offset + span]
                work[start + offset] = left + right
                work[start + offset + span] = left - right
        span *= 2
    return work


def checked_spectral_energy(spectrum: list[int]) -> int:
    require(len(spectrum) == BITS, "unexpected spectrum dimension")
    require(all(-BITS <= coefficient <= BITS for coefficient in spectrum), "spectrum exceeds bipolar bounds")
    return sum(coefficient * coefficient for coefficient in spectrum)


def walsh_inverse_checked(spectrum: list[int]) -> Vector:
    require(len(spectrum) == BITS, "unexpected spectrum dimension")
    require(all(-BITS <= coefficient <= BITS for coefficient in spectrum), "spectrum exceeds bipolar bounds")
    transformed = walsh_in_place(spectrum)
    words = [0] * WORDS
    for index, value in enumerate(transformed):
        require(value % BITS == 0, "inverse coefficient is not integral")
        bipolar = value // BITS
        require(bipolar in (-1, 1), "inverse is not a packed bipolar field")
        if bipolar == 1:
            word, bit = divmod(index, WORD_BITS)
            words[word] |= 1 << bit
    return tuple(words)


def field_fingerprint(field: Vector) -> int:
    fingerprint = 0
    for word in field:
        fingerprint = rotl64(fingerprint, 7) ^ word
    return fingerprint & MASK64


def run_reference_properties() -> tuple[bytes, Vector, Vector, list[int]]:
    fixture = bytes((0, 255, 1, 128, 64, 192, 10, 0))
    reordered = bytes((0, 10, 192, 64, 128, 1, 255, 0))
    order, distribution = encode_raw_bytes(fixture)
    order_repeat, distribution_repeat = encode_raw_bytes(fixture)
    reordered_order, reordered_distribution = encode_raw_bytes(reordered)

    require(order == order_repeat and distribution == distribution_repeat, "byte-event encoding is not deterministic")
    require(distribution == reordered_distribution, "distribution field must ignore raw event order")
    require(order != reordered_order, "tested raw event reorder did not change order field")
    require(
        field_fingerprint(order) != field_fingerprint(reordered_order),
        "tested raw event reorder did not change diagnostic fingerprint",
    )

    spectrum = walsh_forward(order)
    require(checked_spectral_energy(spectrum) == EXPECTED_ENERGY, "Parseval energy mismatch")
    require(checked_spectral_energy(spectrum) - EXPECTED_ENERGY == 0, "Parseval residual is nonzero")
    require(walsh_inverse_checked(spectrum) == order, "Walsh inverse did not reconstruct packed field")

    positive = tuple(MASK64 for _ in range(WORDS))
    positive_spectrum = walsh_forward(positive)
    require(
        positive_spectrum[0] == BITS and all(value == 0 for value in positive_spectrum[1:]),
        "positive constant field does not have exact Walsh DC spectrum",
    )
    require(walsh_inverse_checked(positive_spectrum) == positive, "positive constant inverse mismatch")
    negative = tuple(0 for _ in range(WORDS))
    negative_spectrum = walsh_forward(negative)
    require(
        negative_spectrum[0] == -BITS and all(value == 0 for value in negative_spectrum[1:]),
        "negative constant field does not have exact Walsh DC spectrum",
    )
    require(walsh_inverse_checked(negative_spectrum) == negative, "negative constant inverse mismatch")

    try:
        checked_spectral_energy([BITS + 1] + spectrum[1:])
    except AssertionError:
        pass
    else:
        raise AssertionError("out-of-range spectrum was accepted")
    try:
        walsh_inverse_checked([0] * BITS)
    except AssertionError:
        pass
    else:
        raise AssertionError("non-bipolar bounded inverse was accepted")

    return fixture, order, distribution, spectrum


def parse_report(output: str) -> dict[str, str]:
    report: dict[str, str] = {}
    for raw_line in output.splitlines():
        if ': ' not in raw_line:
            continue
        key, value = raw_line.split(': ', maxsplit=1)
        require(key not in report, f"duplicate CLI report key {key!r}")
        report[key] = value
    return report


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def verify_cli(field_binary: Path, fixture: bytes, order: Vector, distribution: Vector) -> None:
    require(field_binary.is_file(), f"missing field binary: {field_binary}")
    with tempfile.TemporaryDirectory(prefix="vegai-tokenless-field-") as temporary_name:
        temporary = Path(temporary_name)
        fixture_path = temporary / "raw-octets.bin"
        fixture_path.write_bytes(fixture)
        first = run_command([str(field_binary), "inspect", str(fixture_path)])
        require(first.returncode == 0, f"field inspection failed:\n{first.stdout}")
        second = run_command([str(field_binary), "inspect", str(fixture_path)])
        require(second.returncode == 0, f"repeat field inspection failed:\n{second.stdout}")
        require(first.stdout == second.stdout, "field CLI report is not byte-for-byte deterministic")

        report = parse_report(first.stdout)
        expected = {
            "schema": "vegai-tokenless-signal-field-inspection/v1",
            "ingestion": "raw-octet-event/no-vocabulary",
            "learned_tokenizer_used": "false",
            "raw_input_lossless_in_field": "false",
            "raw_source_identity_requirement": "external-digest-and-provenance",
            "bytes": str(len(fixture)),
            "field_dimension_bits": str(BITS),
            "max_input_bytes": str(MAX_BYTES),
            "order_field_fingerprint_xor64": f"{field_fingerprint(order):016X}",
            "distribution_field_fingerprint_xor64": f"{field_fingerprint(distribution):016X}",
            "order_walsh_energy": str(EXPECTED_ENERGY),
            "order_parseval_expected_energy": str(EXPECTED_ENERGY),
            "order_parseval_residual": "0",
            "order_inverse_exact": "true",
            "scope": "finite representation invariants only; no semantic or physical-world claim",
        }
        require(report == expected, f"unexpected field CLI report:\n{first.stdout}")

        reordered_path = temporary / "reordered-octets.bin"
        reordered_path.write_bytes(bytes((0, 10, 192, 64, 128, 1, 255, 0)))
        reordered = run_command([str(field_binary), "inspect", str(reordered_path)])
        require(reordered.returncode == 0, f"reordered inspection failed:\n{reordered.stdout}")
        reordered_report = parse_report(reordered.stdout)
        require(
            reordered_report["distribution_field_fingerprint_xor64"]
            == report["distribution_field_fingerprint_xor64"],
            "CLI distribution field changed for a byte reorder",
        )
        require(
            reordered_report["order_field_fingerprint_xor64"]
            != report["order_field_fingerprint_xor64"],
            "CLI order field did not change for tested byte reorder",
        )

        missing = run_command([str(field_binary), "inspect", str(temporary / "missing.bin")])
        require(missing.returncode != 0 and "status -704" in missing.stdout, "missing source did not fail closed")

        empty_path = temporary / "empty.bin"
        empty_path.write_bytes(b"")
        empty = run_command([str(field_binary), "inspect", str(empty_path)])
        require(empty.returncode != 0 and "status -701" in empty.stdout, "empty source did not fail closed")

        oversized_path = temporary / "oversized.bin"
        oversized_path.write_bytes(b"\x00" * (MAX_BYTES + 1))
        oversized = run_command([str(field_binary), "inspect", str(oversized_path)])
        require(oversized.returncode != 0 and "status -702" in oversized.stdout, "oversized source did not fail closed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--field-bin",
        type=Path,
        help="optional compiled vegai_field executable for end-to-end contract coverage",
    )
    arguments = parser.parse_args()

    try:
        fixture, order, distribution, _ = run_reference_properties()
        if arguments.field_bin is not None:
            verify_cli(arguments.field_bin.resolve(), fixture, order, distribution)
    except (AssertionError, OSError, ValueError) as error:
        print(f"Tokenless Signal Field verification: FAIL: {error}", file=sys.stderr)
        return 1

    if arguments.field_bin is None:
        print("Tokenless Signal Field algebra oracle: PASS")
    else:
        print("Tokenless Signal Field algebra + CLI integration: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
