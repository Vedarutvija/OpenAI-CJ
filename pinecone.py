import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session

app = Flask(__name__)
app.secret_key = "3db1083617e63bb0f7d9fbda2020cd8e"  # Change this to a random secret key
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = ""
# Path to the existing SQLite database
db_path = "chroma_vectors.db"

# Create embeddings
embeddings = OpenAIEmbeddings()

# Load the Chroma vectorstore from the SQLite database
docsearch = Chroma(persist_directory=db_path, embedding_function=embeddings)

# Define routes
@app.route("/")
def index():
    return render_template("base.html")
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    message = data.get('message')
    


# Create ConversationalRetrievalChain with the loaded vectors
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4o"),
    retriever=docsearch.as_retriever(search_kwargs={"k": 1}),
)

# Example usage with a conversational query
chat_history = []

while True:
    query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        break

    result = chain({"question": query, "chat_history": chat_history})
    print(result['answer'])

    chat_history.append((query, result['answer']))
