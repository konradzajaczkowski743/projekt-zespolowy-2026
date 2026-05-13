import requests
import json
import re
import os

def ask(name):
    prompt = (
        f"What is the approximate height of {name} in meters? "
        f"Respond ONLY with a single integer representing the height in meters. "
        f"Do not add any text, units, or explanation."
    )
    try:
        response = requests.post(
            "http://gemma:11434/api/generate",
            json={
                "model": "gemma4:e2b",
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        text = response.json().get("response", "").strip()
        print(f"Name '{name}' -> Raw response: '{text}'")
        match = re.search(r"(\d+)", text)
        if match:
            print(f"   Parsed: {match.group(1)}")
        else:
            print("   Parsed: None")
    except Exception as e:
        print(f"Error for {name}: {e}")

ask("la sagrada familia")
ask("Sagrada Familia")
