import vertexai
from vertexai.language_models import ChatModel

# 01. Set an Environments
PROJECT_ID = ""
REGION = ""
vertexai.init(project=PROJECT_ID, location=REGION)


# 02. Choose the Model
MODEL_NAME = "chat-bison@001"
chat_model = ChatModel.from_pretrained(MODEL_NAME)
chat = chat_model.start_chat(
    temperature=0.1,
    max_output_tokens=1024,
    top_p=0.8,
    top_k=40,

)

# 03. Input Message
# Prompts: "The opposite of hot is"
input_msg = input("\nYou: ")

# 04. Request with Message
chat_response=chat.send_message(input_msg)

# 05. Response
print("\nPaLM 2:", chat_response.text)


# 06. The features of PaLM Vertex AI
# Response has many metadats such as text, satefyAttributes, etc.
print("\nTrue Response:")
print(chat_response)