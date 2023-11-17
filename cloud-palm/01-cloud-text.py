import vertexai
from vertexai.language_models import TextGenerationModel, \
                                     TextEmbeddingModel, \
                                     ChatModel, \
                                     InputOutputTextPair, \
                                     CodeGenerationModel, \
                                     CodeChatModel


# 01. Set an Environments
PROJECT_ID = ""
REGION = ""
vertexai.init(project=PROJECT_ID, location=REGION)


# 02. Choose the Model
MODEL_NAME = "text-bison@latest"
generation_model = TextGenerationModel.from_pretrained(MODEL_NAME)


# 03. Input Prompt
# Prompts: "The opposite of hot is"
input_word = input("\nPrompts: ")

# 04. Request with Prompt
response = generation_model.predict(
    prompt=input_word,
    max_output_tokens=1024,
    temperature=0.1,
    top_k=40,
    top_p=0.95,
    )

# 04. Response PaLM API
# Response: cold.
print("\nResponse:"+response.text)

# 05. The features of PaLM Vertex AI
# Response has many metadats such as text, satefyAttributes, etc.
print("\nTrue Response:")
print(response)