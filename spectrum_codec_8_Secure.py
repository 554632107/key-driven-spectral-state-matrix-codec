"""8-state spectral codec: plain map + key-secure map."""

import numpy as np
from typing import Tuple, Optional
import hashlib
import hmac

# Names match the 8 xlsx class files.
STATE_NAMES = [
    "365ex",
    "980ex",
    "1550ex",
    "365_980ex",
    "365_1550ex",
    "980ex_tm1",
    "365_980ex_tm2",
    "365ex_tm3",
]


class SpectrumCodec8:
    """Fixed 3-bit → 8-state table. No key."""

    def __init__(self, num_states: int = 8):
        if num_states != 8:
            raise ValueError("This codec supports 8 states only.")

        self.num_states = num_states
        self.bits_per_cell = 3
        self.state_names = list(STATE_NAMES)
        self.encode_map = {
            "000": 0,
            "001": 1,
            "010": 2,
            "011": 3,
            "100": 4,
            "101": 5,
            "110": 6,
            "111": 7,
        }
        self.decode_map = {v: k for k, v in self.encode_map.items()}

        print(f"SpectrumCodec8: {num_states} states, {self.bits_per_cell} bit/cell")
        print(f"   states: {', '.join(self.state_names)}")

    def encode_message(self, message: str, matrix_size: int = 10) -> Tuple[np.ndarray, int]:
        """Text → UTF-8 bits → 3-bit groups → state matrix."""
        binary = "".join(format(byte, "08b") for byte in message.encode("utf-8"))
        original_bits = len(binary)

        total_cells = matrix_size * matrix_size
        max_bits = total_cells * self.bits_per_cell
        print(f"[encode] {matrix_size}x{matrix_size}, capacity {max_bits} bit")

        if original_bits > max_bits:
            needed_size = int(np.ceil(np.sqrt(original_bits / self.bits_per_cell)))
            raise ValueError(
                f"Message needs {original_bits} bit, matrix holds {max_bits} bit. "
                f"Use at least {needed_size}x{needed_size}."
            )

        padding_len = (3 - original_bits % 3) % 3
        binary += "0" * padding_len

        matrix_flat = []
        for i in range(0, len(binary), 3):
            matrix_flat.append(self.encode_map.get(binary[i : i + 3], 0))
        while len(matrix_flat) < total_cells:
            matrix_flat.append(0)

        matrix = np.array(matrix_flat[:total_cells]).reshape(matrix_size, matrix_size)
        print(f"[encode] {len(message)} chars → {original_bits} bit → {matrix_size}x{matrix_size}")
        return matrix, original_bits

    def decode_matrix(self, predictions: np.ndarray, original_bits: Optional[int] = None) -> str:
        """State matrix → 3-bit groups → UTF-8 text."""
        if original_bits is None:
            raise ValueError("original_bits is required.")

        binary = ""
        invalid_count = 0
        for pred in predictions.flatten():
            pred = int(pred)
            if pred in self.decode_map:
                binary += self.decode_map[pred]
            else:
                binary += "000"
                invalid_count += 1

        if invalid_count:
            print(f"[decode] {invalid_count} unknown states treated as 000")

        binary = binary[:original_bits]
        message_bytes = [
            int(binary[i : i + 8], 2) for i in range(0, len(binary), 8) if len(binary[i : i + 8]) == 8
        ]
        try:
            message = bytes(message_bytes).decode("utf-8", errors="ignore")
        except Exception:
            message = "decode failed"

        print(f"[decode] {original_bits} bit → '{message}'")
        return message

    def calculate_capacity(self, matrix_size: int) -> dict:
        """Bit/byte capacity of an N×N matrix."""
        total_cells = matrix_size * matrix_size
        total_bits = total_cells * self.bits_per_cell
        total_bytes = total_bits // 8
        return {
            "matrix_size": f"{matrix_size}x{matrix_size}",
            "total_cells": total_cells,
            "total_bits": total_bits,
            "total_bytes": total_bytes,
            "max_chars_ascii": total_bytes,
            "max_chars_utf8": total_bytes // 3,
        }


class SpectrumCodec8Secure:
    """Key permutes the 3-bit table and scrambles cell positions.

    Uses SHA-256 + HMAC. Custom scheme, not equivalent to AES.
    """

    def __init__(self, key: str, num_states: int = 8):
        if num_states != 8:
            raise ValueError("SpectrumCodec8Secure supports 8 states only.")

        self.num_states = num_states
        self.bits_per_cell = 3
        self.state_names = list(STATE_NAMES)

        self._master_key = hashlib.sha256(key.encode("utf-8")).digest()
        self._state_key = hmac.new(
            self._master_key, b"SPECTRUM_STATE_PERMUTATION_V1", hashlib.sha256
        ).digest()
        self._position_key = hmac.new(
            self._master_key, b"SPECTRUM_POSITION_PERMUTATION_V1", hashlib.sha256
        ).digest()

        bits_list = ["000", "001", "010", "011", "100", "101", "110", "111"]
        perm_states = self._generate_secure_permutation(self._state_key, num_states)
        self.encode_map = {bits: int(st) for bits, st in zip(bits_list, perm_states)}
        self.decode_map = {int(st): bits for bits, st in self.encode_map.items()}

        print(f"SpectrumCodec8Secure key='{key}'")
        print(f"   state permutation: {perm_states}")

    def _generate_secure_permutation(self, key_material: bytes, length: int) -> np.ndarray:
        """HMAC Fisher–Yates shuffle."""
        indices = list(range(length))
        for i in range(length - 1, 0, -1):
            rand_material = hmac.new(key_material, i.to_bytes(4, "little"), hashlib.sha256).digest()
            j = int.from_bytes(rand_material[:4], "little") % (i + 1)
            indices[i], indices[j] = indices[j], indices[i]
        return np.array(indices)

    def _get_position_permutation(self, matrix_size: int) -> np.ndarray:
        """Position shuffle for a given matrix size."""
        size_specific_key = hmac.new(
            self._position_key,
            f"SIZE_{matrix_size}x{matrix_size}".encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return self._generate_secure_permutation(size_specific_key, matrix_size * matrix_size)

    def _get_inverse_permutation(self, perm: np.ndarray) -> np.ndarray:
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(len(perm))
        return inv_perm

    def encode_message(self, message: str, matrix_size: int = 10) -> Tuple[np.ndarray, int]:
        """Text → states in logical order → position scramble → matrix."""
        utf8_bytes = message.encode("utf-8")
        binary = "".join(format(byte, "08b") for byte in utf8_bytes)
        original_bits = len(binary)

        total_cells = matrix_size * matrix_size
        max_bits = total_cells * self.bits_per_cell
        print(f"[secure-encode] '{message}'  {matrix_size}x{matrix_size}, {original_bits} bit")

        if original_bits > max_bits:
            needed_size = int(np.ceil(np.sqrt(original_bits / self.bits_per_cell)))
            raise ValueError(
                f"Message needs {original_bits} bit, matrix holds {max_bits} bit. "
                f"Use at least {needed_size}x{needed_size}."
            )

        padding_len = (3 - original_bits % 3) % 3
        binary += "0" * padding_len

        logical_states = []
        for i in range(0, len(binary), 3):
            logical_states.append(self.encode_map.get(binary[i : i + 3], 0))
        while len(logical_states) < total_cells:
            logical_states.append(0)

        logical_states = np.array(logical_states, dtype=int)
        perm = self._get_position_permutation(matrix_size)
        matrix = logical_states[perm].reshape(matrix_size, matrix_size)
        return matrix, original_bits

    def decode_matrix(self, predictions: np.ndarray, original_bits: Optional[int] = None) -> str:
        """Unscramble positions, then invert the key-dependent 3-bit table."""
        if original_bits is None:
            raise ValueError("original_bits is required.")

        matrix_size = predictions.shape[0]
        physical_flat = predictions.flatten().astype(int)
        if len(physical_flat) != matrix_size * matrix_size:
            raise ValueError("Prediction matrix size mismatch.")

        perm = self._get_position_permutation(matrix_size)
        logical_states = physical_flat[self._get_inverse_permutation(perm)]

        bits_list = []
        invalid_count = 0
        for st in logical_states:
            st = int(st)
            if st in self.decode_map:
                bits_list.append(self.decode_map[st])
            else:
                bits_list.append("000")
                invalid_count += 1

        if invalid_count:
            print(f"[secure-decode] {invalid_count} unknown states treated as 000")

        binary = "".join(bits_list)[:original_bits]
        message_bytes = [
            int(binary[i : i + 8], 2) for i in range(0, len(binary), 8) if len(binary[i : i + 8]) == 8
        ]
        try:
            message = bytes(message_bytes).decode("utf-8", errors="ignore")
        except Exception:
            message = "decode failed"

        print(f"[secure-decode] '{message}'")
        return message


if __name__ == "__main__":
    print("=" * 70)
    print("8-state codec self-test")
    print("=" * 70)

    codec = SpectrumCodec8()
    print("\n[capacity]")
    for size in [5, 10, 20, 30]:
        cap = codec.calculate_capacity(size)
        print(f"{cap['matrix_size']}: {cap['total_bits']} bit ({cap['total_bytes']} bytes)")

    for msg in ["HELLO", "Multi-Wavelength Spectral Encoding", "光谱编码测试"]:
        print(f"\n{'=' * 70}\nplain: '{msg}'")
        encoded, original_bits = codec.encode_message(msg, matrix_size=10)
        decoded = codec.decode_matrix(encoded, original_bits=original_bits)
        print("ok" if decoded == msg else f"mismatch: '{decoded}'")

    print("\n" + "=" * 70)
    print("8 states = 3 bit/cell. 10x10 holds 300 bit (37 bytes).")
    print("=" * 70)
