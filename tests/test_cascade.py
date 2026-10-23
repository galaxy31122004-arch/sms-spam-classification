import unittest

from tier1.cascade import route_message


RULES = {
    "thresholds": {"spam_confident": 1.0, "ham_confident": -1.0},
    "allowlist": {"domains": ["google.com", "apple.com"]},
    "force_to_tier2": [
        {
            "name": "url_la",
            "type": "url_not_allowlisted",
            "reason": "untrusted link",
        },
        {
            "name": "trung_thuong_vay_viec",
            "type": "keyword_with_contact",
            "keywords": ["win", "prize", "loan", "job", "offer"],
            "reason": "suspicious offer",
        },
        {
            "name": "tin_nhan_rut_gon",
            "type": "shortlink",
            "domains": ["bit.ly", "t.co", "tinyurl.com"],
            "reason": "short link",
        },
        {
            "name": "so_dien_thoai_la",
            "type": "unknown_sender_phone",
            "reason": "unknown sender with phone",
        },
    ],
}


class CascadeRoutingTests(unittest.TestCase):
    def test_confident_scores_are_handled_by_tier1(self):
        spam = route_message("ordinary text", 1.2, RULES)
        ham = route_message("ordinary text", -1.2, RULES)

        self.assertEqual((spam.tier, spam.label), ("tier1", "spam"))
        self.assertEqual((ham.tier, ham.label), ("tier1", "ham"))

    def test_untrusted_url_overrides_confident_ham_route(self):
        decision = route_message("Your account is locked https://odd.example", -2.0, RULES)

        self.assertEqual(decision.tier, "tier2")
        self.assertEqual(decision.rule, "url_la")

    def test_allowlisted_url_does_not_force_tier2(self):
        decision = route_message("Visit https://mail.google.com", -2.0, RULES)

        self.assertEqual((decision.tier, decision.label), ("tier1", "ham"))

    def test_contact_keyword_with_phone_forces_tier2(self):
        decision = route_message("You win! Call 09061234567", 2.0, RULES)

        self.assertEqual((decision.tier, decision.rule), ("tier2", "trung_thuong_vay_viec"))

    def test_unknown_sender_rule_only_applies_when_contact_state_is_false(self):
        text = "Call 09061234567"

        unknown = route_message(text, -2.0, RULES)
        known = route_message(text, -2.0, RULES, sender_in_contacts=True)
        unknown_contact = route_message(text, -2.0, RULES, sender_in_contacts=False)

        self.assertEqual(unknown.tier, "tier1")
        self.assertEqual(known.tier, "tier1")
        self.assertEqual((unknown_contact.tier, unknown_contact.rule), ("tier2", "so_dien_thoai_la"))


if __name__ == "__main__":
    unittest.main()
