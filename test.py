import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI , GoogleGenerativeAIEmbeddings
from langchain.document_loaders import TextLoader


# Load .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

#load document
loader = TextLoader("faq/account.txt", encoding="utf-8")
docs = loader.load()


# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)


#Embeddings (models/embedding-001)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector = embeddings.embed_query("what is my name?")
print(str(vector))
