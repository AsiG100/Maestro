import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm.client import LLMClient, OpenAILLMClient


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(text="mock response")])


class FakeOpenAI:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.completions = FakeCompletions()


class OpenAILLMClientTest(unittest.TestCase):
    def test_initializes_client_from_environment_and_formats_prompt(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-api-key",
                "OPENAI_BASE_URL": "https://example.invalid/v1",
            },
            clear=True,
        ):
            with patch("llm.client.OpenAI", FakeOpenAI):
                client = OpenAILLMClient(model="gpt-test")
                result = client.completations(
                    user_question="What is a concerto?",
                    user_type="kid",
                )

        self.assertEqual(result, "mock response")
        self.assertEqual(client.client.api_key, "test-api-key")
        self.assertEqual(client.client.base_url, "https://example.invalid/v1")
        self.assertEqual(len(client.client.completions.calls), 1)
        self.assertEqual(
            client.client.completions.calls[0],
            {
                "model": "gpt-test",
                "prompt": (
                    "Use kid to determine the best way you can answer the question.\n"
                    "    If the user is a kid, use simpler language and explain things in a way that is easy to understand.\n"
                    "    If the user is a parent, use more complex language and explain things in a way that is easy to understand.\n"
                    "    Avoid highly technical language and jargon.\n"
                    "    \n"
                    "    User question:\n"
                    "    What is a concerto?\n"
                    "    "
                ),
                "max_tokens": 1000,
            },
        )

    def test_llm_client_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            LLMClient()


if __name__ == "__main__":
    unittest.main()
