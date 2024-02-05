import requests
from langchain_core.embeddings import Embeddings

class APIEmbeddings(Embeddings):
    def __init__(self, API_URL):
        self.API_URL = API_URL

    def embed_documents(self, texts):
        payload = {"inputs": texts}
        response = requests.post(self.API_URL, json=payload)
        return response.json()

    def embed_query(self, text):
        payload = {"inputs": [text]}
        response = requests.post(self.API_URL, json=payload)
        return response.json()[0]