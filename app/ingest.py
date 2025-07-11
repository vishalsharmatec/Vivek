import os
import glob
import pandas as pd
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader, DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import GoogleGenerativeAIEmbeddings

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Load and combine documents
docs = []

# Load all .txt files
for path in glob.glob("FAQ/*.txt"):
    loader = TextLoader(path)
    docs.extend(loader.load())

# Load .parquet and convert to documents
df = pd.read_parquet("data/User_detail_report.parquet")
df_loader = DataFrameLoader(df, page_content_column="your_column_name")  # Replace this
docs.extend(df_loader.load())

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# Embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
db = Chroma.from_documents(split_docs, embeddings, persist_directory="chroma_db")
db.persist()
