from dotenv import load_dotenv
load_dotenv()

import csv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI

# Define the prompt
prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="""
Given the following question and its answer, generate 10 diverse but similar questions that would have the same answer.
Make sure the questions are logically different but all clearly have the same correct answer.

Original Question: {question}
Answer: {answer}

10 Similar Questions:
"""
)

llm = OpenAI(temperature=0.7, max_tokens=1000)
chain = LLMChain(llm=llm, prompt=prompt)

# Load CSV
csv_path = "output.csv"  
output_lines = []

with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for idx, row in enumerate(reader):
        question = row['question']
        answer = row['answer']
        print(f"Generating for row {idx + 1}...")

        try:
            generated = chain.run({"question": question, "answer": answer})
            output_lines.append(
                f"Original Question: {question}\nAnswer: {answer}\nGenerated:\n{generated}\n\n"
            )
        except Exception as e:
            output_lines.append(f"Error generating for row {idx + 1}: {e}\n\n")

# Save to output file
with open("generated_similar_questions.txt", "w", encoding="utf-8") as f:
    f.writelines(output_lines)

print("Done! Results saved to 'generated_similar_questions.txt'.")
