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


# 02. Model List
# Response:
#   models/chat-bison-001
#   models/text-bison-001
#   models/text-bison-recitation-off
#   models/text-bison-safety-off
#   models/text-bison-safety-recitation-off
#   models/embedding-gecko-001
#   models/embedding-gecko-002
for model in palm.list_models():
    print(model.name)