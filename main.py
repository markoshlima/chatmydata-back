import os
import sys
import json

import datetime
import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

import constants

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)

@app.route('/data')
@cross_origin()
def get_data():
  result = ""
  with open('data/data.json') as user_file:
    result = json.loads(user_file.read())
  return result

@app.route('/history')
@cross_origin()
def get_history():
  return data_history

@app.route('/chat', methods=['POST'])
@cross_origin()
def do_chat():
  input = request.get_json(force=True) 
  question = input['question']
  result = chain({"question": question, "chat_history": chat_history})
  answer = result['answer']
  chat_history.append((question, answer))
  history = {
    "user": question,
    "bot": answer,
    "time": datetime.datetime.now()
  }
  add_data_history(history)  
  return history

def load_data_history():
  data_history.append({
      "user": "Who is the oldest?",
      "bot": "The oldest person is Dintano, who is 56 years old.",
      "time": datetime.datetime.now()
  })
  data_history.append({
      "user": "Quem é o mais novo?",
      "bot": "O mais novo é o Ciclano, com 13 anos de idade.",
      "time": datetime.datetime.now()
  })
  data_history.append({
      "user": "Qual a soma de todas as idades?",
      "bot": "A soma de todas as idades é 168 anos.",
      "time": datetime.datetime.now()
  })

def add_data_history(history):
  data_history.append(history)

os.environ["OPENAI_API_KEY"] = constants.APIKEY
chat_history = []
data_history = []
load_data_history()
text_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader("data/", glob="**/*.json", show_progress=True, loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
docs = loader.load()
print(docs)
index = VectorstoreIndexCreator().from_loaders([loader])
chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)