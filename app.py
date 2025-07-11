import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI , GoogleGenerativeAIEmbeddings


# Load .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

#load document

#Embeddings (models/embedding-001)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
embeddings.embed_query

llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3) # gemini-2.0-flash-001
result = llm.invoke("Write me a ballad about LangChain")

print(result.content)
 

# splitted_text = RecursiveCharacterTextSplitter()