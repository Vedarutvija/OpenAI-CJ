from openai import OpenAI
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
import sys
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

os.environ["OPENAI_API_KEY"]="sk-cloudjune-H0RLPEiyk8tHQri0KPArT3BlbkFJM61NxbpwCOhnOJVflWQP"
query = None
if len(sys.argv)>1:
    query = sys.argv[1]

loader = DirectoryLoader("data/")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
db_path = "chroma_vectors.db"
docsearch = Chroma.from_documents(texts, embeddings, persist_directory=db_path)
# docsearch = Chroma.from_documents(texts, embeddings)

# qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

# chain = ConversationalRetrievalChain.from_llm(
#     llm=ChatOpenAI(model="gpt-4o"),
#     retriever=docsearch.as_retriever(search_kwargs={"k":1}),
# )

# chat_history = []

# while True:
#     if not query:
#         query = input("Prompt: ")
#     if query in ['quit', 'q', 'exit']:
#         sys.exit()
#     result = chain({"question": query, "chat_history": chat_history})
#     print(result['answer'])

#     chat_history.append((query, result['answer']))
#     query = None


