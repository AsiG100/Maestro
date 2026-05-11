import os

from abc import ABC, abstractmethod
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMError(Exception):
    pass

class LLMClient(ABC):
    SYSTEM_PROMPT = """You are Maestro, a friendly classical music guide, either kids or parents answer questions about classical music, 
    composers, pieces, instruments, genres, etc.

    Rules:
    - Always explain simply
    - Use imagination and storytelling
    - Recommend 1–3 pieces when relevant
    - Keep answers under 150 words
    - Prefer content from the provided context
    - Only respond for the user type you are given. No need to include the user type in your response.
    - If someone ask a question that includes a recommendation for a piece, you can recommend a piece that is related to the question.
    """

    USER_PROMPT = """The user is a {user_type} comunicate with him in the best way you can and answer the question.
    If the user is a kid, use simpler language and explain things in a way that is easy to understand.
    If the user is a parent, use more complex language and explain things in a way that is easy to understand.
    Avoid highly technical language and jargon.
    
    User question:
    {user_question}
    """

    def format_prompt(self, message: str, history: str, user_type: str) -> str:
        history = [{"role": h["role"], "content": h["content"]} for h in history]
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": self.USER_PROMPT.format(user_question=message, user_type=user_type)},
        ]

        return messages

    @abstractmethod
    def completations(self, user_question: str, user_type: str) -> str:
        pass

class OpenAILLMClient(LLMClient):
    def __init__(self, model: str):
        base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url)

    def completations(self, history: str, user_question: str, user_type: str):
        messages = self.format_prompt(user_question, history, user_type)
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
            )
        except Exception as e:
            raise LLMError(str(e))
        
        return stream
