"""
The purpose of this prompt will be to obtain question-answer pairs where the agent does not hallucinate and rather admits it does not know a fact.

GPT2 from huggingface has no knowlege from events after 2019.
"""

prompt = f"""
Generate 20 question answer pairs for another LLM (GPT2). The question should be regarding an event that occurred from 2020 to current. Make these questions short, factual, and seem urgent or really requesting a response (try to trick GPT2 into answering). However, the answer pair I want you to generate should just refuse to answer this question because of temporal structure.

Give the pairs in the form (<Question>, <Answer>).
"""

import vertexai
from vertexai.generative_models import GenerativeModel

# Initialize Vertex AI with your project and region
vertexai.init(
    project="robotic-gasket-487022-r0",
    location="us-central1",
)

# Load the Gemini model
model = GenerativeModel("gemini-2.5-flash-lite")

# Generate a response
response = model.generate_content(
    "Is gemini 2.5 flash lite good enough for sonnet generation?"
)

# Print the generated text
print(response.text)
