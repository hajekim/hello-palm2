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
# Prompts: "The opposite of hot is"
input_word = input("\nPrompts: ")

# 03. Request with Prompt
response = palm.generate_text(
    model=MODEL_NAME,
    prompt=input_word,
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