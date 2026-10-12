from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

# Setting the environment
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="growing_vegetables")

# Documents loader

loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()

textSplitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function = len,
    is_separator_regex = False,
)


chunks = textSplitter.split_documents(raw_documents)

#Prepare Chroma DB arguments to upload data.

documents = []
metadata = []
ids = []

for i,chunk in enumerate(chunks):
    ids.append("ID"+str(i))
    metadata.append(chunk.metadata)
    documents.append(chunk.page_content)


# Put Data in ChromaDB Collection

collection.upsert(
    ids= ids,
    metadatas=metadata,
    documents=documents,
)