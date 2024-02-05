import os
import pandas as pd
import time
import PyPDF2
import anthropic
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceBgeEmbeddings, SentenceTransformerEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders.excel import UnstructuredExcelLoader
from langchain.schema.document import Document
from unstructured.cleaners.core import clean_extra_whitespace
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import common_functions as cf
from APIEmbeddings import *

user_path = '/Users/tammyauyong/Documents/Projects/bot-alex/index-creation/'
policies_path = '/Users/tammyauyong/Documents/Projects/bot-alex/index-creation/Policies/'

def split_dataframe(dataframe, chunk_size):
    num_chunks = len(dataframe) // chunk_size + (1 if len(dataframe) % chunk_size != 0 else 0)
    chunks = [dataframe.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    return chunks

def while_replace(string):
    while '  ' in string:
        string = string.replace('  ', ' ')
    return string
    
def obtain_document_info(file):
    df_doc_info = pd.read_csv(user_path+'file_mapping.csv', index_col=False)
    doc_info_dict = json.loads(df_doc_info.to_json(orient="records", index=False))
    try:
        (url, last_modified) = next((doc_info['url'], doc_info['modified_time']) 
                                    for doc_info in doc_info_dict if file == doc_info['name'])
    except StopIteration:
        (url, last_modified) = (None, None)
    return [url, last_modified]

def create_index_embeddings(folder_path, index_name):
    main_path = '/Users/tammyauyong/Documents/Projects/bot-alex/index-creation/index/'
    API_URL = "http://dpo.asuscomm.com:8088/predict"
    embedding = APIEmbeddings(API_URL)
    #embedding = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en")
    finalised_documents = []
    for path, subdirs, files in os.walk(folder_path):
        for name in files:
            file = os.path.join(path, name)
            folder_name = os.path.basename(path)
            file_url_source = obtain_document_info(name)[0]
            file_last_modified = obtain_document_info(name)[1]
            
            if file.endswith(('.pdf','.docx','.pptx')):
                loader = UnstructuredFileLoader(file_path=file,mode='paged',post_processors=[clean_extra_whitespace])
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(separators=['\n'], chunk_size=2048, chunk_overlap=256)
                docs = text_splitter.split_documents(documents)
                count=0
                for doc in docs:
                    doc.metadata['source'] = file_url_source
                    doc.metadata['last_modified'] = file_last_modified
                    doc.metadata['file_chunk_index'] = count
                    doc.metadata['type'] = folder_name
                    count += 1
                    try:
                        page = doc.metadata['page_number']
                    except:
                        doc.metadata['page_number'] = 1
                    finalised_documents.append(doc)
            
            elif file.endswith(('.xlsx', '.xls', '.csv', '.xlsm')):
                documents = json.loads(cf.retrieve_multi_excel_text([file]))
                file_url_source = obtain_document_info(name)[0]
                file_last_modified = obtain_document_info(name)[1]
                count = 0
                for sheet_name, json_docs in documents.items():
                    for json_doc in json_docs:
                        doc = Document(page_content=json.dumps(json_doc), 
                                       metadata= {'filename':name
                                                  ,'page_number': sheet_name
                                                  ,'source': file_url_source
                                                  ,'last_modified': file_last_modified
                                                  ,'file_chunk_index': count
                                                  ,'type': folder_name
                                                  })
                        count += 1
                        finalised_documents.append(doc)
            
    processed_files_w_url = []
    for doc in finalised_documents:
        processed_files_w_url.append([doc.metadata['filename'], doc.metadata['page_number'], doc.metadata['source'], doc.metadata['last_modified']])
    df_final_documents = pd.DataFrame(processed_files_w_url)
    df_final_documents.to_csv(main_path+f'{index_name}_processed_files.csv', index=False)

    vectordb = FAISS.from_documents(documents=finalised_documents, embedding=embedding)
    vectordb.save_local(folder_path=main_path, index_name=f"{index_name}")
    return f"Index Embedding <<{index_name}>> created successfully"

#create index embeddings
#create_index_embeddings(policies_path+'gl', "fin_hr_procurement_BGE_gl")
#create_index_embeddings(policies_path+'finance',"fin_hr_procurement_BGE_finance")
create_index_embeddings(policies_path+'procurement',"fin_hr_procurement_BGE_procurement")