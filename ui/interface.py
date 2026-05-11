from typing import Callable
from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel

from llm.client import OpenAILLMClient

class ChatEngine(BaseModel):
    fn: Callable


class UserInterface(ABC):
    interface_engine: ChatEngine

    def chat(self, message: str, history: List[str]):
        history = '\n'.join([str({"role": h["role"], "content": h["content"]}) for h in history])
   
        return message + history + 'from base user interface!'

    @abstractmethod
    def launch(self):
        pass

class ChatUserInterface(UserInterface):
    def __init__(self, interface_engine: ChatEngine, user_type: str):
        self.interface_engine = interface_engine(
            fn=self.chat,
            title="Maestro",
            description="A chatbot powered by Popsical",
            examples=[
                ["What are examples of classical music pieces?"],
                ["What are the main genres of classical music?"],
                ["What are the main composers of classical music?"],
                ["How can I learn more about classical music?"],
                ["Where should I start listening to classical music?"],
                ["What are the main instruments of classical music?"],
                ["What should my child listen to before bed?"],
                ["Who is Mozart?"],
                ["Give me a fun classical music fact!"],
            ],
            flagging_mode="never",
        )
        self.llm_client = OpenAILLMClient(model="gpt-4o-mini")
        self.user_type = user_type
    
    def handle_tool_calls(self, message: dict) -> List[dict]:
        return [{"role": "assistant", "content": message.content}]

    def chat(self, message: str, history: str):
        
        stream_response = self.llm_client.completations(
            history=history,
            user_question=message,
            user_type=self.user_type,
        )

        # while response.finish_reason == "tool_calls":
        #     msg = response.message
        #     tool_responses = self.handle_tool_calls(msg)
        #     response = openai.chat.completions.create(
        #         model=MODEL_CLAUDE,
        #         messages=messages,
        #         tools=tools,
        #     )

        result = ""
        for chunk in stream_response:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="")
                result += content
                yield result
    
    def launch(self):
        self.interface_engine.launch(theme="origin")
