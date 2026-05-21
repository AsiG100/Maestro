import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

MUSIC_INTENT_KEYWORDS = [
    "music", "playlist", "song", "piece", "listen", "youtube", "spotify",
    "recommend", "recommendation", "classical", "play something", "what should we listen"
]


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()


def load_music_links(path: str = "data/music_links.json") -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def detect_music_intent(user_message: str) -> bool:
    """Return True when the user appears to be asking for listening recommendations."""
    text = normalize_text(user_message)
    return any(keyword in text for keyword in MUSIC_INTENT_KEYWORDS)


def detect_mood(user_message: str, music_data: Dict[str, Any]) -> Optional[str]:
    """Return the best matching mood key from music_links.json, or None."""
    text = normalize_text(user_message)

    best_mood = None
    best_score = 0

    for mood_key, mood_data in music_data.get("moods", {}).items():
        score = 0
        for keyword in mood_data.get("intent_keywords", []):
            keyword_normalized = normalize_text(keyword)
            if keyword_normalized and keyword_normalized in text:
                score += 1

        if score > best_score:
            best_score = score
            best_mood = mood_key

    return best_mood


def get_music_recommendations(user_message: str, path: str = "data/music_links.json") -> Optional[Dict[str, Any]]:
    """
    Tool-style function for chatbot use.

    Returns:
      - None when no music intent is detected.
      - A mood-specific recommendation payload when intent and mood match.
      - A fallback recommendation payload when music intent exists but no mood is clear.
    """
    music_data = load_music_links(path)

    if not detect_music_intent(user_message):
        return None

    mood_key = detect_mood(user_message, music_data)

    if mood_key:
        mood_data = music_data["moods"][mood_key]
        return {
            "matched_intent": "music_recommendation",
            "matched_mood": mood_key,
            "display_name": mood_data["display_name"],
            "description": mood_data["description"],
            "recommendations": mood_data["recommendations"]
        }

    fallback = music_data["fallback"]
    return {
        "matched_intent": "music_recommendation",
        "matched_mood": "fallback",
        "display_name": fallback["display_name"],
        "description": fallback["description"],
        "recommendations": fallback["recommendations"]
    }


if __name__ == "__main__":
    test_messages: List[str] = [
        "Can you give me a bedtime classical playlist for my child?",
        "What should we listen to that feels magical?",
        "Do you have something fun for dancing?",
        "Tell me about Beethoven."
    ]

    for message in test_messages:
        print("USER:", message)
        print(json.dumps(get_music_recommendations(message), indent=2, ensure_ascii=False))
        print("-" * 80)
