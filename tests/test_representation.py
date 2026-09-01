from types import SimpleNamespace
import unittest
from unittest.mock import patch

from phishing_detection.representation import (
    ACTIVE_URL_RE,
    deidentify,
    enrich_v2,
    parse_email,
)


class RepresentationTests(unittest.TestCase):
    def test_plain_text_is_preferred_over_html(self):
        raw = (
            b"Content-Type: multipart/alternative; boundary=x\n\n"
            b"--x\nContent-Type: text/plain\n\nPlain body\n"
            b"--x\nContent-Type: text/html\n\n<b>HTML body</b>\n--x--"
        )
        parsed = parse_email(raw)
        self.assertEqual(parsed.body, "Plain body")
        self.assertEqual(parsed.body_source, "plain")

    def test_html_fallback_omits_hidden_content(self):
        parsed = parse_email(
            b"Content-Type: text/html\n\n<p>Visible</p><script>secret()</script>"
        )
        self.assertEqual(parsed.body, "Visible")

    def test_deidentification_removes_active_values(self):
        fake_document = SimpleNamespace(ents=[])
        with patch(
            "phishing_detection.representation._ner",
            return_value=lambda text: fake_document,
        ):
            subject, body, detector = deidentify(
                "Contact alice@example.com",
                "Visit https://bad.example/login?id=12345678",
                sender_domain="example.com",
                recipient_domains=(),
            )
        self.assertNotIn("alice@example.com", subject)
        self.assertNotIn("bad.example", body)
        self.assertIsNone(ACTIVE_URL_RE.search(detector))

    def test_v2_adds_only_inactive_structure_and_counts_generator(self):
        raw = b"Content-Type: text/html\n\n<a href='https://bad.example/login?q=1'>Sign in</a>"
        value = enrich_v2("Subject: Test\n\nBody", raw, (item for item in [".pdf"]))
        self.assertIn("HTML link count: 1", value)
        self.assertIn("Attachment count: 1", value)
        self.assertIn("[ATTACHMENT_EXT_PDF]", value)
        self.assertNotIn("bad.example", value)


if __name__ == "__main__":
    unittest.main()
