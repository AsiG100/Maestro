import os

from abc import ABC, abstractmethod
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv

from rag.client import RagClient

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
    - If someone asks about popsical's concerts, offer to provide a link to the upcoming concerts page

    For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
    {context}
    """

    USER_PROMPT = """The user is a {user_type} comunicate with him in the best way you can and answer the question.
    If the user is a kid, use simpler language and explain things in a way that is easy to understand.
    If the user is a parent, use more complex language and explain things in a way that is easy to understand.
    Avoid highly technical language and jargon.
    
    User question:
    {user_question}
    """

    def __init__(self, model: str):
        self.model = model
        self.rag = RagClient(self.model)

    def _format_prompt(self, system_prompt: str, user_prompt: str, history: str) -> List[Dict]:
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": user_prompt},
        ]

        return messages
    
    def _construct_messages(self, question: str, history: str, user_type: str) -> List[Dict]:
        chunks = self.rag.fetch_context(question, history)
        context = "\n\n".join(
            f"Data type is {chunk.metadata['type']}:\n{chunk.page_content}" for chunk in chunks
        )
        user_prompt = self.USER_PROMPT.format(user_question=question, user_type=user_type)
        system_prompt = self.SYSTEM_PROMPT.format(context=context)
        history = [{"role": h["role"], "content": h["content"]} for h in history]
        return self._format_prompt(system_prompt, user_prompt, history)

    @abstractmethod
    def completations(self, user_question: str, user_type: str) -> str:
        pass

class OpenAILLMClient(LLMClient):
    def __init__(self, model: str):
        super().__init__(model)
        base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url)

    def completations(self, history: str, user_question: str, user_type: str):
        messages = self._construct_messages(user_question, history, user_type)
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
            )
        except Exception as e:
            raise LLMError(str(e))
        
        return stream
