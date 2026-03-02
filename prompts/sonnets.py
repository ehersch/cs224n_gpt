"""
The purpose of this prompt will be to prompt a larger model (Gemini 3) to produce in-distribition sonnet data.

Note, our sonnets dataset includes Shakespearean sonnets.
"""

from os import access

import vertexai
from vertexai.generative_models import GenerativeModel
from tqdm import tqdm


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


if __name__ == "__main__":
    model_spec = "gemini-2.5-flash-lite"
    project = "robotic-gasket-487022-r0"
    location = "us-central1"

    for start_idx in tqdm(range(661, 1000, 20), desc="Generating sonnets"):
        p = prompt(start_idx)
        repsonse = access_llm(p, project, location, model_spec)

        path = "synthetic_data/synthetic_sonnets.txt"
        save_response(repsonse, path)
