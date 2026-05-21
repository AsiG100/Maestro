import json
from logging import Logger
from typing import Dict, List
from agents.agent import Agent, AgentError
from agents.recommendation.recommendation_agent import RecommendationAgent

class PlanningAgent(Agent):
    MODEL = "openai/gpt-4.1-nano"

    def __init__(self):
        self.log("Planning Agent is initializing")
        self.recommendation = RecommendationAgent()
        self.memory = None
        self.opportunity = None
        self.log("Planning Agent is ready")
    
    def recommend_playlist(self, playlist_name: str = 'fallback') -> str:
        """
        Run the tool to recommend a playlist
        """
        self.log("Planning agent is calling recommendation")
        result = self.recommendation.recommend(playlist_name)
        return result.model_dump_json()

    def get_tools(self):
            """
            Return the json for the tools to be used
            """
            return [
                {"type": "function", "function": self.recommendation.make_function_template()},
            ]
        
    def handle_tool_call(self, tool_calls: List) -> List[Dict]:
        """
        Actually call the tools associated with this message
        """
        mapping = {
            "recommend_playlist": self.recommend_playlist,
        }
        results = []
        try:
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                tool = mapping.get(tool_name)
                result = tool(**arguments) if tool else ""
                results.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})
            return results
        except Exception as e:
            logger = Logger()
            logger.log(str(e))
            raise AgentError(str(e))
    