import json
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel

from agents.agent import Agent


class Playlist(BaseModel):
    display_name: str
    description: str
    recommendations: List[Dict]
    intent_keywords: Optional[List[str]] = None

class RecommendationAgent(Agent):
    def __init__(self) -> None:
        self.music_links = self.load_music_links()

    @staticmethod
    def load_music_links() -> Dict:
        path = 'agents/recommendation/music_links.json'
        with Path(path).open("r", encoding="utf-8") as file:
            return json.load(file)


    def make_function_template(self):
        return {
            "name": "recommend_playlist",
            "description": "Returns a playlist name from the playlists json file that fits the mood of the user",
            "parameters": {
                "type": "object",
                "properties": {
                        "playlist_name": {
                            "type": "string",
                            "description": "The playlist name out of the list of playlist offered by our knowledge base",
                            "options": list(self.music_links.keys())
                        },
                    },
                "required": ['playlist_name'],
                "additionalProperties": False,
            },
        }

    def recommend(self, playlist_name: str) -> Playlist:
        return Playlist.model_validate(self.music_links[playlist_name])