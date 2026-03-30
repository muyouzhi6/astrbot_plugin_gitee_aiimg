import base64
import unittest

from core.image_format import (
    decode_base64_image_payload,
    guess_image_mime_and_ext_strict,
)


PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+X2ioAAAAASUVORK5CYII="
)


class ImageFormatTests(unittest.TestCase):
    def test_decode_raw_base64_image_payload(self):
        raw = decode_base64_image_payload(PNG_B64)

        self.assertEqual(guess_image_mime_and_ext_strict(raw), ("image/png", "png"))

    def test_decode_data_url_image_payload(self):
        raw = decode_base64_image_payload(f"data:image/png;base64,{PNG_B64}")

        self.assertEqual(guess_image_mime_and_ext_strict(raw), ("image/png", "png"))

    def test_decode_base64_scheme_image_payload(self):
        raw = decode_base64_image_payload(f"base64://{PNG_B64}")

        self.assertEqual(guess_image_mime_and_ext_strict(raw), ("image/png", "png"))

    def test_reject_non_image_payload(self):
        text_b64 = base64.b64encode(b"hello world").decode()

        with self.assertRaises(ValueError):
            decode_base64_image_payload(text_b64)


if __name__ == "__main__":
    unittest.main()
