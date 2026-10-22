#!/usr/bin/env python3
"""Independent, dependency-free reference verification for Vegai HDS algebra.

This is intentionally a Python model rather than a transcription of Fortran
control flow.  It validates the mathematical contracts, deterministic seeds,
exact noise cardinality, and the three end-to-end retrieval workloads used by
the executable demo.  Native Fortran property tests live in test/test_all.f90.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose

BITS = 16_384
WORDS = 256
WORD_BITS = 64
MASK64 = (1 << 64) - 1
SIGNLESS64 = (1 << 63) - 1
GAMMA = 0x9E3779B97F4A7C15
MUL1 = 0xBF58476D1CE4E5B9
MUL2 = 0x94D049BB133111EB
ID_TAG = 0xD1B54A32D192ED03


def rotl64(value: int, amount: int) -> int:
    amount %= 64
    value &= MASK64
    return ((value << amount) | (value >> ((64 - amount) % 64))) & MASK64


def splitmix_next(state: int) -> tuple[int, int]:
    state = (state + GAMMA) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * MUL1) & MASK64
    z = ((z ^ (z >> 27)) * MUL2) & MASK64
    return state, (z ^ (z >> 31)) & MASK64


def seed_from_id(symbol_id: int) -> int:
    low = symbol_id & 0xFFFF_FFFF
    return rotl64(low, 17) ^ ID_TAG ^ rotl64(low, -11)


def from_seed(seed: int) -> tuple[int, ...]:
    state = seed & MASK64
    values: list[int] = []
    for _ in range(WORDS):
        state, value = splitmix_next(state)
        values.append(value)
    return tuple(values)


def bind(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(left ^ right for left, right in zip(a, b, strict=True))


def hamming(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return sum((left ^ right).bit_count() for left, right in zip(a, b, strict=True))


def similarity(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    return 1.0 - 2.0 * hamming(a, b) / BITS


def permute(vector: tuple[int, ...], shift: int) -> tuple[int, ...]:
    shift %= BITS
    if shift == 0:
        return vector
    word_shift, bit_shift = divmod(shift, WORD_BITS)
    result: list[int] = []
    for output_word in range(WORDS):
        source_high = (output_word - word_shift) % WORDS
        if bit_shift == 0:
            result.append(vector[source_high])
        else:
            source_low = (output_word - word_shift - 1) % WORDS
            result.append(
                ((vector[source_high] << bit_shift) & MASK64)
                | (vector[source_low] >> (WORD_BITS - bit_shift))
            )
    return tuple(result)


def bundle(vectors: list[tuple[int, ...]], tie_seed: int) -> tuple[int, ...]:
    assert vectors
    tie = from_seed(tie_seed)
    result = [0] * WORDS
    for word in range(WORDS):
        output = 0
        for bit in range(WORD_BITS):
            count = sum(1 if (vector[word] >> bit) & 1 else -1 for vector in vectors)
            if count > 0 or (count == 0 and ((tie[word] >> bit) & 1)):
                output |= 1 << bit
        result[word] = output
    return tuple(result)


def bundle3(a: tuple[int, ...], b: tuple[int, ...], c: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        (left & middle) | (middle & right) | (left & right)
        for left, middle, right in zip(a, b, c, strict=True)
    )


def noise_exact(vector: tuple[int, ...], flips: int, seed: int) -> tuple[int, ...]:
    assert 0 <= flips <= BITS
    mask = [0] * WORDS
    selected = 0
    attempts = 0
    state = seed & MASK64
    while selected < flips:
        state, random_word = splitmix_next(state)
        position = (random_word & SIGNLESS64) % BITS
        word, bit = divmod(position, WORD_BITS)
        if not ((mask[word] >> bit) & 1):
            mask[word] |= 1 << bit
            selected += 1
        attempts += 1
        if attempts >= BITS * 16 and selected < flips:
            for position in range(BITS):
                word, bit = divmod(position, WORD_BITS)
                if not ((mask[word] >> bit) & 1):
                    mask[word] |= 1 << bit
                    selected += 1
                    if selected == flips:
                        break
            break
    return bind(vector, tuple(mask))


@dataclass
class Memory:
    entries: list[tuple[int, str, tuple[int, ...]]]

    def __init__(self) -> None:
        self.entries = []

    def register(self, symbol_id: int, label: str) -> tuple[int, ...]:
        assert all(existing != symbol_id for existing, _, _ in self.entries)
        vector = from_seed(seed_from_id(symbol_id))
        self.entries.append((symbol_id, label, vector))
        return vector

    def cleanup(self, query: tuple[int, ...]) -> tuple[int, str, float, int]:
        ranked = sorted(
            (hamming(query, vector), position, symbol_id, label)
            for position, (symbol_id, label, vector) in enumerate(self.entries)
        )
        distance, _, symbol_id, label = ranked[0]
        return symbol_id, label, 1.0 - 2.0 * distance / BITS, distance


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_unit_properties() -> None:
    # Official first SplitMix64 output for state zero is a useful fixed oracle.
    state, first = splitmix_next(0)
    require(first == 0xE220A8397B1DCDAF, "SplitMix64 reference oracle changed")
    a = from_seed(12_345)
    b = from_seed(67_890)
    c = from_seed(-7_654_321)
    zero = tuple(0 for _ in range(WORDS))

    require(from_seed(12_345) == a, "seed determinism failed")
    require(7_000 < hamming(a, b) < 9_400, "random vectors are not near orthogonal")
    require(bind(a, zero) == a, "XOR identity failed")
    require(bind(a, bind(a, b)) == b, "XOR unbinding failed")
    require(permute(permute(a, 129), -129) == a, "permutation inverse failed")
    require(permute(a, BITS + 19) == permute(a, 19), "permutation wrap failed")
    require(bundle([a, b, c], 777) == bundle3(a, b, c), "majority equivalence failed")

    noisy = noise_exact(a, 4_096, 44)
    require(hamming(a, noisy) == 4_096, "exact noise cardinality failed")
    require(noisy == noise_exact(a, 4_096, 44), "noise determinism failed")
    require(isclose(similarity(a, a), 1.0), "similarity identity failed")


def encode_frame(roles: list[tuple[int, ...]], fillers: list[tuple[int, ...]]) -> tuple[int, ...]:
    return bundle([bind(role, filler) for role, filler in zip(roles, fillers, strict=True)], 0x5A17F00D3C4B2E19)


def encode_mapping(source: list[tuple[int, ...]], target: list[tuple[int, ...]]) -> tuple[int, ...]:
    return bundle([bind(left, right) for left, right in zip(source, target, strict=True)], 0x6E624EB7D3A91F05)


def encode_causal(causes: list[tuple[int, ...]], effects: list[tuple[int, ...]]) -> tuple[int, ...]:
    return bundle(
        [bind(cause, permute(effect, 1)) for cause, effect in zip(causes, effects, strict=True)],
        0x13C0FFEE7A5E1D42,
    )


def run_end_to_end_demo_contract() -> None:
    memory = Memory()
    role_unit = memory.register(101, "ROLE:PROCESSING_UNIT")
    role_function = memory.register(102, "ROLE:PRIMARY_FUNCTION")
    role_output = memory.register(103, "ROLE:GENERATED_OUTPUT")
    alu = memory.register(201, "HARDWARE:ALU_CORE")
    arithmetic = memory.register(202, "FUNCTION:LOGIC_ARITHMETIC")
    flags = memory.register(203, "OUTPUT:STATUS_FLAGS")
    hippocampus = memory.register(301, "NEURO:HIPPOCAMPUS")
    consolidation = memory.register(302, "FUNCTION:MEMORY_CONSOLIDATION")
    long_memory = memory.register(303, "OUTPUT:EPISODIC_TRACE")
    anomaly = memory.register(401, "EVENT:SENSOR_ANOMALY")
    heat = memory.register(402, "EVENT:CRITICAL_HEAT_RISE")
    pressure = memory.register(403, "EVENT:PRESSURE_SPIKE")
    valve = memory.register(404, "ACTION:OPEN_RELIEF_VALVE")
    shutdown = memory.register(405, "ACTION:REACTOR_SHUTDOWN")
    for symbol_id in range(500, 1_000):
        memory.register(symbol_id, "BACKGROUND_CONCEPT")

    roles = [role_unit, role_function, role_output]
    cpu = [alu, arithmetic, flags]
    brain = [hippocampus, consolidation, long_memory]
    cpu_frame = encode_frame(roles, cpu)
    noisy_cpu_frame = noise_exact(cpu_frame, 4_096, 135_792_468)
    recovered_id, _, recovered_similarity, _ = memory.cleanup(bind(noisy_cpu_frame, role_function))
    require(recovered_id == 202, "25% noisy frame did not recover arithmetic filler")
    require(recovered_similarity > 0.15, "frame retrieval margin unexpectedly weak")

    mapping = encode_mapping(cpu, brain)
    analogy_id, _, analogy_similarity, _ = memory.cleanup(bind(mapping, alu))
    require(analogy_id == 301, "aligned analogy did not recover hippocampus")
    require(analogy_similarity > 0.35, "analogy retrieval margin unexpectedly weak")

    rule_bank = encode_causal([anomaly, heat, pressure, valve], [heat, pressure, valve, shutdown])
    state = anomaly
    for expected in (402, 403, 404, 405):
        probe = permute(bind(rule_bank, state), -1)
        actual, _, confidence, _ = memory.cleanup(probe)
        require(actual == expected, f"causal chain expected {expected}, got {actual}")
        require(confidence > 0.25, "causal retrieval margin unexpectedly weak")
        state = next(vector for symbol_id, _, vector in memory.entries if symbol_id == actual)


def main() -> None:
    run_unit_properties()
    run_end_to_end_demo_contract()
    print("Python reference verification: PASS")


if __name__ == "__main__":
    main()
