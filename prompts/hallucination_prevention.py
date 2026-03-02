"""
The purpose of this prompt will be to obtain question-answer pairs where the agent does not hallucinate and rather admits it does not know a fact.

GPT2 from huggingface has no knowlege from events after 2019.

The 1000 QA pairs cover diverse subjects: half are general across finance, sports, tech, science, etc. Then 200 are related to US sports, 150 are related to current events in major US cities, and the rest are related to events on US college campuses and natural disasters around the world.
"""

from os import access

import vertexai
from vertexai.generative_models import GenerativeModel
from tqdm import tqdm
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
        f.write(response + "\n")


def prompt(batch_id: int, n_pairs: int = 50) -> str:
    prompt = f"""
    You are generating synthetic training data for GPT-2 (knowledge cutoff: 2019).
    Create {n_pairs} (Question, Answer) pairs.

    Constraints:
    - Each Question MUST ask about a real-world event or fact from 2020 through today.
    - Questions should be short, factual, and "urgent"/pressuring (trying to trick the model into answering).
    - Cover the following domain: politics, sports, tech, entertainment, finance, science, world events.
    - DO NOT include any answers that state the fact. The Answer MUST always refuse due to knowledge cutoff.

    Answer format rules:
    - Output exactly one pair per line.
    - Format: (<Question>, <Answer>)
    - The Answer MUST follow this template closely:

    "I don't know. My training data only goes up to 2019, so I can't reliably answer questions about events from 2020 or later."

    Batch id: {batch_id}
    """.strip()
    return prompt


def generate_data():
    model_spec = "gemini-2.5-flash-lite"
    project = "robotic-gasket-487022-r0"
    location = "us-central1"

    for batch_id in tqdm(range(0, 1), desc="Generating hallucination QA"):
        p = prompt(batch_id=batch_id, n_pairs=200)
        repsonse = access_llm(p, project, location, model_spec)
        path = "synthetic_data/anti_hallucination_qa.txt"
        save_response(repsonse, path)


def process_data(path_in, path_out):
    data = []

    with open(path_in, "r") as f:
        for line in f:
            first = line.split("###Question###")[1]
            lst = first.split(", ###Answer###")
            if len(lst) != 2:
                print(lst)

            else:
                question = lst[0]
                answer = lst[1]
                data += [[question, answer]]

    with open(path_out, "w") as json_file:
        # 3. Use json.dump() to write the data to the file
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    # generate_data()
    process_data(
        "synthetic_data/anti_hallucination_qa.txt",
        "synthetic_data/anti_hallucination_qa.json",
    )
