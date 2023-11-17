# 
# This code is using Google PaLM 2 API (Not Vertex AI's one)
# 

import google.generativeai as palm

# 01. Google API Key Setting
# Get your API KEY here:
# https://makersuite.google.com/app/apikey
API_KEY = ""
palm.configure(api_key=API_KEY)

MODEL_NAME="models/text-bison-001"

# 02. Input Prompt
query_word = """
This is my first overseas trip. What should I prepare?
Please recommend a good place to go for a honeymoon.
"""
print("\n",query_word)

# 03. Prompt Template
prompt_template = """
You are an expert at travel agency.
You sincerely answer customer questions.
The reason is also explained in detail.

{query}

Your solution:
"""

# 04. Request with Prompt
response = palm.generate_text(
    model=MODEL_NAME,
    prompt=prompt_template.format(query=query_word),
    temperature=1,
    top_k=5,
    top_p=1,
    max_output_tokens=800
    )

# 04. Response PaLM API
# Response: cold.
print("\nResponse:", response.result)

# 05. True Response
print("\nTrue Response")
print(response)