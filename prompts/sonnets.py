"""
The purpose of this prompt will be to prompt a larger model (Gemini 3) to produce in-distribition sonnet data.

Note, our sonnets dataset includes Shakespearean sonnets.
"""

from os import access

import vertexai
from vertexai.generative_models import GenerativeModel
from tqdm import tqdm
import re
import json


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
    model_spec = "gemini-2.5-flash"
    project = "robotic-gasket-487022-r0"
    location = "us-central1"

    for start_idx in tqdm(range(0, 1000, 20), desc="Generating sonnets"):
        p = prompt(start_idx + 1)
        repsonse = access_llm(p, project, location, model_spec)

        path = "synthetic_data/synthetic_sonnets_2.txt"
        save_response(repsonse, path)


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
    # generate_data()
    # process_sonnets("synthetic_data/synthetic_sonnets_2.txt")

    ## This is the final format
    txt = """My love is not a star, though bright she gleams,\nNor like the sun, whose fiery course doth range,\nBut constant as the moon in silver dreams,\nAbove the shifting world of earthly change.\nLet envious clouds obscure the morning's light,\nAnd fickle fortune turn her wheel about,\nMy faith in her shall stand through darkest night,\nUnmoved by slander, malice, fear, or doubt.\nFor true affection knows no ebb nor flow,\nBut holds its course, steadfast in every trial,\nA rooted oak, where winter tempests blow,\nIt scorns deceit, and shuns the tongue of guile.\n\tThus fixed in truth, my heart shall ever be,\n\tAs constant as her love, eternally.
    """
    # print(txt)
