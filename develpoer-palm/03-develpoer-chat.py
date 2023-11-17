# 
# This code is using Google PaLM 2 API (Not Vertex AI's one)
# 

import google.generativeai as palm

# 01. Google API Key Setting
# Get your API KEY here:
# https://makersuite.google.com/app/apikey
API_KEY = ""
palm.configure(api_key=API_KEY)

# 02. Input Message
input_msg = input("\nYou: ")

# 03. Request Message
chat_response = palm.chat(messages=input_msg)

# 04. Reponse Chat
print("\nPaLM 2:", chat_response.last)

# 05. Print every response
print("\nTrue Response")
print(chat_response)