
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
Generate a CSV table with 3 rows about: {topic}
- Use commas to separate values
- Wrap all text fields in double quotes (even if they don't contain commas)
- Do NOT use markdown or pipes
- Only return raw CSV (no explanation or code block)

Example format:
"Column1","Column2","Column3"
"Row1Col1","Row1Col2","Row1Col3"

Now generate the table:
"""
)

llm = OpenAI(temperature=0.7)
chain = LLMChain(llm=llm, prompt=prompt)

result = chain.run("programming languages")
# Save the CSV result to a file
with open("output2.csv", "w", encoding="utf-8") as f:
    f.write(result)

print("CSV file saved as output.csv ✅")
