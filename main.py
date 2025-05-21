
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
Generate a dataset of 50 question-answer pairs about {topic}.
- Each question should be clear, concise, and related to programming languages (syntax, use cases, comparisons, history, etc.).
- Each answer should be short, factual, and beginner-friendly (1–2 sentences).
- Output ONLY valid CSV format (comma-separated values) with the following headers:
"Question","Answer"
- Wrap all text fields in double quotes to ensure CSV compatibility.
- Do NOT include code fences, markdown, or extra explanations.

Example:
"Question","Answer"
"What is Python used for?","Python is used for web development, data science, automation, and AI."
"What is the difference between Java and JavaScript?","Java is a general-purpose programming language, while JavaScript is mainly used for web development."

Now generate the 50 questions and answers.
"""
)


llm = OpenAI(temperature=0.3, max_tokens=3000)
chain = LLMChain(llm=llm, prompt=prompt)

result = chain.run("programming languages")
# Save the CSV result to a file
with open("output.csv", "w", encoding="utf-8") as f:
    f.write(result)

print("CSV file saved as output.csv ✅")
