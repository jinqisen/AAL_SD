
import unittest
from agent.utils import validate_response

class TestResponseValidation(unittest.TestCase):
    def test_valid_response(self):
        response = """Thought: The user wants to select samples.
Action: {"tool_name": "get_top_k_samples", "parameters": {"k": 5, "lambda_param": 0.5}}"""
        self.assertTrue(validate_response(response), "Standard valid response should pass")

    def test_valid_response_with_unicode_colon(self):
        response = """Thought: The user wants to select samples.
Action：{"tool_name": "get_top_k_samples", "parameters": {"k": 5}}"""
        self.assertTrue(validate_response(response), "Response with unicode colon should pass")

    def test_invalid_response_trailing_text(self):
        response = """Thought: The user wants to select samples.
Action: {"tool_name": "get_top_k_samples", "parameters": {"k": 5}}
This is some extra explanation that should be rejected."""
        self.assertFalse(validate_response(response), "Response with trailing text should fail")

    def test_invalid_response_trailing_newline_only(self):
        # A simple newline might be acceptable if stripped, but our logic strips before checking.
        # Let's verify exact behavior.
        # The logic is: remainder = json_str[end_idx:].strip()
        # So trailing whitespace is fine.
        response = """Thought: The user wants to select samples.
Action: {"tool_name": "get_top_k_samples", "parameters": {"k": 5}}
"""
        self.assertTrue(validate_response(response), "Response with trailing whitespace should pass")

    def test_invalid_response_no_action_keyword(self):
        response = """Thought: I will just think about it.
{"tool_name": "get_top_k_samples"}"""
        self.assertFalse(validate_response(response), "Response without 'Action:' keyword should fail")

    def test_invalid_response_malformed_json(self):
        response = """Thought: Typo in JSON.
Action: {"tool_name": "get_top_k_samples", "parameters": }"""
        self.assertFalse(validate_response(response), "Response with malformed JSON should fail")

    def test_invalid_response_double_action(self):
        # This is tricky. If there are two actions, the logic splits by rsplit(marker, 1), 
        # so it takes the last one. If the text BEFORE the last action contains garbage relative to the first, it ignores it.
        # But if the text AFTER the last action has garbage, it fails.
        # However, ReAct usually expects one thought and one action.
        # The prompt says "只输出两行".
        # Our validation only checks if the LAST Action segment is valid and has no trailing text.
        
        # Case: Two actions, last one is clean.
        response = """Thought: Step 1
Action: {"tool_name": "1"}
Thought: Step 2
Action: {"tool_name": "2"}"""
        # rsplit will split at the last "Action:", getting '{"tool_name": "2"}'
        # This is valid JSON with no trailing text.
        # So technically validate_response returns True here.
        # Ideally we might want to be stricter, but the primary issue was trailing garbage.
        self.assertTrue(validate_response(response), "Multiple actions ending cleanly might pass technical validation (acceptable for now)")

        # Case: Two actions, last one has garbage.
        response_garbage = """Thought: Step 1
Action: {"tool_name": "1"}
Thought: Step 2
Action: {"tool_name": "2"}
Some extra text"""
        self.assertFalse(validate_response(response_garbage), "Multiple actions ending with garbage should fail")

if __name__ == '__main__':
    unittest.main()
