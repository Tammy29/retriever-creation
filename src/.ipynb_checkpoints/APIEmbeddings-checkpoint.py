import requests
from langchain_core.embeddings import Embeddings

class APIEmbeddings(Embeddings):
    def __init__(self, API_URL):
        self.API_URL = API_URL
        self.query_instruction = 'Represent this sentence for searching relevant passages: '

    def embed_documents(self, texts):
        payload = {"inputs": texts}
        response = requests.post(self.API_URL, json=payload)
        return response.json()

    def embed_query(self, text):
        text = self.query_instruction + text.replace('\n', ' ')
        payload = {"inputs": text}
        response = requests.post(self.API_URL, json=payload)
        return response.json()