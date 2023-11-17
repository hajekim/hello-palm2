import vertexai
from vertexai.language_models import TextEmbeddingModel
import numpy as np


# 01. Set an Environments
PROJECT_ID = ""
REGION = ""
vertexai.init(project=PROJECT_ID, location=REGION)

# Function: Get Vector Embedding Value from Response
def getValue(embedding_value):
    for embedding in embedding_value:
        vector = embedding.values
        return vector

# Query
query = 'What do squirrels eat?'
close_to_query = 'nuts and acorns'
different_from_query = 'This morning I woke up in San Francisco, and took a walk to the Bay Bridge. It was a good, sunny morning with no fog.'

print("\nQuery:",query)
print("\nRight Answer:",close_to_query)
print("\nFail Answer:",different_from_query)

# 02. Choose the Model
MODEL_NAME = "textembedding-gecko@001"
embedding_model = TextEmbeddingModel.from_pretrained(MODEL_NAME)
embeddings = embedding_model.get_embeddings


# 03. Create vector embeddings
# Prompts: "The opposite of hot is"
embedding_query = embedding_model.get_embeddings([query])
embedding_close_to_query = embedding_model.get_embeddings([close_to_query])
embedding_different_from_query = embedding_model.get_embeddings([different_from_query])
    
print("\n01. Vector Embedding")
print(getValue(embedding_query))

similar_measure = np.dot(getValue(embedding_query), getValue(embedding_close_to_query))
print("\n02. 쿼리와 관계가 높은 답변과의 내적")
print(similar_measure)


different_measure = np.dot(getValue(embedding_query), getValue(embedding_different_from_query))
print("\n03. 쿼리와 관계가 적은 답변과의 내적")
print(different_measure)


# 04. True Response
print("\nTrue Response:")
print(embedding_query)