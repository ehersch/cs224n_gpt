"""
The purpose of this prompt will be to prompt a larger model (Gemini 3) to produce in-distribition sonnet data.

Note, our sonnets dataset includes Shakespearean sonnets.
"""

from os import access
import os

import vertexai
from vertexai.generative_models import GenerativeModel
import asyncio
import threading
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import json

load_dotenv(override=True)


def access_llm(prompt, project, location, model_spec):
    vertexai.init(
        project=project,
        location=location,
    )

    # Load the Gemini model
    model = GenerativeModel(model_spec)

    # Generate a response
    response = model.generate_content(prompt)
    return response.text


def access_llm_openrouter(prompt, model="google/gemini-2.5-pro"):
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8192,
    )
    return response.choices[0].message.content


def save_response(response, path):
    with open(path, "a") as f:  # append to end
        f.write(response + "\n" + "\n")


def prompt(starting_index: int) -> str:
    return f"""
    Generate 20 Shakespearean sonnets in English, in the style of Shakespeare's original sonnets.

    Hard constraints for EACH sonnet:
    - Exactly 14 non-empty lines.
    - No title line.
    - Strict end-rhyme scheme: ABAB CDCD EFEF GG (clear end rhymes; avoid slant/near rhymes).
    - Shakespearean diction and syntax (Early Modern English feel; no modern slang).

    Formatting constraints:
    - Before each sonnet, output a single line containing exactly: ###{starting_index}### for the first sonnet, then increment by 1 for each next sonnet.
    - After the delimiter, output the 14 lines of the sonnet.
    - Output nothing else.

    Begin.
    """.strip()


def generate_data():
    model_spec = "gemini-2.5-pro"
    project = "robotic-gasket-487022-r0"
    location = "us-central1"

    for start_idx in tqdm(range(0, 1000, 20), desc="Generating sonnets"):
        p = prompt(start_idx + 1)
        repsonse = access_llm(p, project, location, model_spec)

        path = "synthetic_data/synthetic_sonnets_pro.txt"
        save_response(repsonse, path)


def _fetch_batch(start_idx):
    """Fetch a single batch of 20 sonnets. Returns (start_idx, response_text)."""
    p = prompt(start_idx + 1)
    response = access_llm_openrouter(p, model="google/gemini-2.5-pro")
    return start_idx, response


def generate_data_openrouter(max_workers=10):
    path = "synthetic_data/synthetic_sonnets_pro.txt"
    write_lock = threading.Lock()
    batches = list(range(0, 1000, 20))  # 50 batches

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_batch, idx): idx for idx in batches}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating sonnets (OpenRouter)"):
            start_idx, response = future.result()
            with write_lock:
                save_response(response, path)


def process_sonnets(path):
    """
    Ensures last 2 lines indented (not sure if important)
    Saves them as JSON list
    """
    with open(path, "r") as f:
        text = f.read()

    lst = re.split("###.*###", text)
    sonnets = []
    for sonnet in lst[1:]:
        # split and remove blank lines
        lines = [l.strip() for l in sonnet.split("\n") if l.strip()]

        if len(lines) >= 14:
            lines[-2] = "\t" + lines[-2]
            lines[-1] = "\t" + lines[-1]

        sonnets.append("\n".join(lines))

    with open("synthetic_data/sonnets_flash.json", "w") as f:
        json.dump(sonnets, f, indent=2)


if __name__ == "__main__":
    generate_data_openrouter()
    # process_sonnets("synthetic_data/synthetic_sonnets_2.txt")

    ## This is the final format
    txt = """My love is not a star, though bright she gleams,\nNor like the sun, whose fiery course doth range,\nBut constant as the moon in silver dreams,\nAbove the shifting world of earthly change.\nLet envious clouds obscure the morning's light,\nAnd fickle fortune turn her wheel about,\nMy faith in her shall stand through darkest night,\nUnmoved by slander, malice, fear, or doubt.\nFor true affection knows no ebb nor flow,\nBut holds its course, steadfast in every trial,\nA rooted oak, where winter tempests blow,\nIt scorns deceit, and shuns the tongue of guile.\n\tThus fixed in truth, my heart shall ever be,\n\tAs constant as her love, eternally.
    """
    # print(txt)
