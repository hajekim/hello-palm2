# 
# This code is using Google PaLM 2 API (Not Vertex AI's one)
# 

import google.generativeai as palm
import numpy as np

# 01. Google API Key Setting
# Get your API KEY here:
# https://makersuite.google.com/app/apikey
API_KEY = ""
palm.configure(api_key=API_KEY)


query = 'What do squirrels eat?'
close_to_query = 'nuts and acorns'
different_from_query = 'This morning I woke up in San Francisco, and took a walk to the Bay Bridge. It was a good, sunny morning with no fog.'

print("\nQuery:",query)
print("\nRight Answer:",close_to_query)
print("\nFail Answer:",different_from_query)

# Create vector embeddings
MODEL_NAME = "models/embedding-gecko-001"
embedding_query = palm.generate_embeddings(model=MODEL_NAME, text=query)
embedding_close_to_query = palm.generate_embeddings(model=MODEL_NAME, text=close_to_query)
embedding_different_from_query = palm.generate_embeddings(model=MODEL_NAME, text=different_from_query)

print("\n01. Vector Embedding")
print(embedding_query)

similar_measure = np.dot(embedding_query['embedding'], embedding_close_to_query['embedding'])
print("\n02. 쿼리와 관계가 높은 답변과의 내적")
print(similar_measure)


different_measure = np.dot(embedding_query['embedding'], embedding_different_from_query['embedding'])
print("\n03. 쿼리와 관계가 적은 답변과의 내적")
print(different_measure)