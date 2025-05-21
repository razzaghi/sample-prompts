from dotenv import load_dotenv
load_dotenv()

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI


# Define prompt
prompt = PromptTemplate(
    input_variables=["article"],
    template="""
Read the following article carefully. Then, generate 10 diverse and meaningful questions based on the content.
The questions should cover key facts, main ideas, and important details.

Article:
{article}

Questions:
"""
)

llm = OpenAI(temperature=0.5, max_tokens=1000)
chain = LLMChain(llm=llm, prompt=prompt)

# --- Load article ---
file_path = "article.txt" 
with open(file_path, "r", encoding="utf-8") as f:
    article_text = f.read()


result = chain.run(article_text)


print("\nGenerated Questions:\n")
print(result)
with open("generated_questions.txt", "w", encoding="utf-8") as f:
    f.write(result)
