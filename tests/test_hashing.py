"""Pinned digests: saved task configs depend on this encoding."""

from pathlib import Path
import tempfile
import unittest

from ufactory_mujoco._hashing import canonical_json_sha256, file_sha256


class HashingTests(unittest.TestCase):
    def test_canonical_json_vector(self):
        self.assertEqual(
            canonical_json_sha256({"b": 1, "a": [1.5, "é", None, True]}),
            "4310c259a6f903aaf441de115a0b417227ea158ea1613ca86eff57e82894d4ba",
        )

    def test_key_order_does_not_matter(self):
        self.assertEqual(canonical_json_sha256({"x": 1, "y": 2}), canonical_json_sha256({"y": 2, "x": 1}))

    def test_file_digest_spans_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blob.bin"
            path.write_bytes(b"\x00" * (3 * 1024 * 1024 + 7))
            self.assertEqual(
                file_sha256(path),
                "f395f662db5cb6d4af68e95e95bf3a780950e8855cd43c7140367d36d6b5d1a3",
            )


if __name__ == "__main__":
    unittest.main()
