"""
gemini_commentary.py

Sends the context JSON (built by context_builder.py) to the Gemini API
with a cricket-commentator system prompt, and returns natural-language
commentary text.

SETUP:
1. Get a free API key from https://aistudio.google.com/apikey
2. Set it as an environment variable (do NOT hardcode it in this file
   or commit it to git):

   PowerShell (temporary, current session only):
       $env:GEMINI_API_KEY = "your-key-here"

   PowerShell (permanent, persists across sessions):
       [System.Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "your-key-here", "User")
       (close and reopen your terminal after running this)

3. Verify it's set:
       echo $env:GEMINI_API_KEY
"""

import os
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_builder import build_context

# load GEMINI_API_KEY from a .env file in the project root, if present
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

SYSTEM_PROMPT = """You are an experienced, engaging cricket commentator calling a live match.
You will be given a JSON object describing a single delivery, including:
- shot_type: the shot played (e.g. "cover_drive"), or null if not confidently identified
- shot_confidence: how confident the system is (0 to 1)
- shot_type_known: whether the shot type should be named in your commentary

Your task: write 1-2 sentences of natural, enthusiastic cricket commentary for this delivery.

RULES:
- Do NOT invent or mention any player names, team names, scores, or over numbers -
  none of that information is provided, so do not make it up. Describe only the shot itself.
- If shot_type_known is true, name the shot naturally (e.g. "That's a gorgeous cover drive!").
- If shot_type_known is false, describe the shot generically based on the event alone
  (e.g. "He gets bat on that and sends it through the off side!") - do NOT guess or name
  a specific shot type.
- Keep it natural and broadcast-style, not robotic. Vary your phrasing - don't reuse
  the same sentence structure every time.
- Do not mention confidence scores, JSON, or any technical details in your output -
  only the commentary text itself.
- Keep it to 1-2 sentences.
"""


def get_model():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable not set. "
            "See the setup instructions at the top of this file."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-3.6-flash",
        system_instruction=SYSTEM_PROMPT,
    )


def generate_commentary(context: dict) -> str:
    model = get_model()
    prompt = json.dumps(context, indent=2)
    response = model.generate_content(prompt)
    return response.text.strip()


if __name__ == "__main__":
    # quick end-to-end test using a sample reconstruction error
    test_error = 0.008  # a "typical" cover-drive-like error, per Day 4 stats
    context = build_context(test_error)
    print("Context sent to Gemini:")
    print(json.dumps(context, indent=2))

    print("\nGenerated commentary:")
    commentary = generate_commentary(context)
    print(commentary)