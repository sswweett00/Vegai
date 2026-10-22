#!/usr/bin/env python3
"""Independent, dependency-free reference verification for Vegai HDS algebra.

This is intentionally a Python model rather than a transcription of Fortran
control flow.  It validates the mathematical contracts, deterministic seeds,
exact noise cardinality, Resonator meet-in-the-middle certification, and the
end-to-end retrieval workloads used by the executable demo. Native Fortran
property tests live in test/test_all.f90.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from itertools import product
from math import isclose

from corpus_manifest import (
    DEFAULT_MAX_LINE_BYTES,
    DEFAULT_MAX_TOKENS_PER_LINE,
    DEFAULT_TOKEN_LENGTH,
    scan_corpus,
)

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


Candidate = tuple[int, str, tuple[int, ...]]


def compose(vectors: list[tuple[int, ...]]) -> tuple[int, ...]:
    result = tuple(0 for _ in range(WORDS))
    for vector in vectors:
        result = bind(result, vector)
    return result


def resonator_fingerprint(vector: tuple[int, ...]) -> int:
    """Same non-cryptographic index fingerprint as the Fortran verifier."""
    return (
        vector[0]
        ^ rotl64(vector[66], 13)
        ^ rotl64(vector[130], -21)
        ^ rotl64(vector[192], 37)
        ^ vector[-1]
    ) & MASK64


def exact_factorize_mitm(
    target: tuple[int, ...], codebooks: list[list[Candidate]], max_half_combinations: int
) -> tuple[tuple[int, ...] | None, int]:
    """Bounded MITM verifier with fingerprint chains and full-vector checks.

    The solution count is intentionally saturated at two, matching the public
    Fortran ambiguity metadata contract.
    """
    require(len(codebooks) >= 2, "factorization requires two or more codebooks")
    split = len(codebooks) // 2
    left_books, right_books = codebooks[:split], codebooks[split:]
    left_size = 1
    right_size = 1
    for book in left_books:
        left_size *= len(book)
    for book in right_books:
        right_size *= len(book)
    require(left_size <= max_half_combinations, "left half exceeds configured verifier bound")
    require(right_size <= max_half_combinations, "right half exceeds configured verifier bound")

    # The dictionary value is a collision chain. Equality is checked below.
    right_index: dict[int, list[tuple[tuple[int, ...], tuple[int, ...]]]] = {}
    for slots in product(*(range(len(book)) for book in right_books)):
        vector = compose([right_books[index][slot][2] for index, slot in enumerate(slots)])
        right_index.setdefault(resonator_fingerprint(vector), []).append((slots, vector))

    first_solution: tuple[int, ...] | None = None
    solution_count = 0
    for left_slots in product(*(range(len(book)) for book in left_books)):
        left_vector = compose([left_books[index][slot][2] for index, slot in enumerate(left_slots)])
        wanted_right = bind(target, left_vector)
        for right_slots, actual_right in right_index.get(resonator_fingerprint(wanted_right), []):
            # This full equality check is the collision-safety contract.
            if wanted_right == actual_right:
                solution_count = min(2, solution_count + 1)
                if first_solution is None:
                    first_solution = tuple(left_slots + right_slots)
                if solution_count == 2:
                    return first_solution, solution_count
    return first_solution, solution_count


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

    # The fourth executable workload: slot-specific Resonator factorization.
    factor_books: list[list[Candidate]] = [
        [(201, "F1:ALU", alu), (301, "F1:HIPPOCAMPUS", hippocampus), (401, "F1:ANOMALY", anomaly)],
        [(202, "F2:ARITHMETIC", arithmetic), (302, "F2:CONSOLIDATION", consolidation), (402, "F2:HEAT", heat)],
        [(203, "F3:FLAGS", flags), (303, "F3:TRACE", long_memory), (403, "F3:PRESSURE", pressure)],
    ]
    resonator_target = compose([hippocampus, heat, flags])
    slots, solution_count = exact_factorize_mitm(resonator_target, factor_books, 64)
    require(slots is not None and solution_count == 1, "MITM verifier did not find unique tuple")
    recovered_ids = tuple(factor_books[index][slot][0] for index, slot in enumerate(slots))
    require(recovered_ids == (301, 402, 203), "Resonator verifier recovered wrong factor IDs")
    recovered_vectors = [factor_books[index][slot][2] for index, slot in enumerate(slots)]
    require(compose(recovered_vectors) == resonator_target, "Resonator factors do not rebind to target")

    # Duplicate codewords are semantically ambiguous and must not be hidden.
    duplicate_a = from_seed(777)
    duplicate_b = from_seed(888)
    duplicate_c = from_seed(999)
    ambiguous_books: list[list[Candidate]] = [
        [(9001, "DUPLICATE:A1", duplicate_a), (9002, "DUPLICATE:A2", duplicate_a)],
        [(9101, "UNIQUE:B", duplicate_b)],
        [(9201, "UNIQUE:C", duplicate_c)],
    ]
    ambiguous_target = compose([duplicate_a, duplicate_b, duplicate_c])
    _, ambiguous_count = exact_factorize_mitm(ambiguous_target, ambiguous_books, 8)
    require(ambiguous_count == 2, "duplicate factor codewords must report ambiguity")


def run_corpus_manifest_contract() -> None:
    """Exercise the bounded local preflight independently of Fortran runtime."""
    fixture = b"HELLO can't\n" + (b"z" * 49) + b" valid\n" + (b"y" * 8193) + b"\n" + b"\xff\n"
    digest, byte_count, stats = scan_corpus(
        BytesIO(fixture),
        DEFAULT_MAX_LINE_BYTES,
        DEFAULT_TOKEN_LENGTH,
        DEFAULT_MAX_TOKENS_PER_LINE,
    )
    require(len(digest) == 64 and byte_count == len(fixture), "manifest SHA-256 scan is malformed")
    require(stats.records == 4, "manifest physical record count is wrong")
    require(stats.accepted_tokens == 3, "manifest accepted-token count is wrong")
    require(stats.dropped_tokens == 1, "manifest long-token drop count is wrong")
    require(stats.records_with_tokenizer_truncation == 1, "manifest truncation record count is wrong")
    require(stats.overlong_records == 1, "manifest overlong-record count is wrong")
    require(stats.maximum_record_bytes == 8193, "manifest maximum-record count is wrong")
    require(stats.non_ascii_bytes == 1, "manifest non-ASCII byte count is wrong")


def main() -> None:
    run_unit_properties()
    run_end_to_end_demo_contract()
    run_corpus_manifest_contract()
    print("Python reference verification: PASS")


if __name__ == "__main__":
    main()
