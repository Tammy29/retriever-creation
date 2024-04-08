import os
import pandas as pd
import time
import PyPDF2
import anthropic
import json
import common_functions as cf
import uuid
import pickle
from langchain.storage import InMemoryStore
from langchain.indexes import VectorstoreIndexCreator
from langchain.retrievers import BM25Retriever
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders.excel import UnstructuredExcelLoader
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.cleaners.core import clean_extra_whitespace
#from APIEmbeddings import *


#user_path = '/Users/tammyauyong/Documents/Projects/bot-alex/index-creation/'
#policies_path = '/Users/tammyauyong/Documents/Projects/bot-alex/index-creation/Policies/'
user_path = '/retriever-creation/'
reference_path = user_path + 'reference/'

def save_object(obj, filename):
    retriever_path = user_path + '/retriever/'
    with open(retriever_path+f'{filename}.pkl', 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def split_dataframe(dataframe, chunk_size):
    num_chunks = len(dataframe) // chunk_size + (1 if len(dataframe) % chunk_size != 0 else 0)
    chunks = [dataframe.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    return chunks

def while_replace(string):
    while '  ' in string:
        string = string.replace('  ', ' ')
    return string
    
def obtain_document_info(file):
    df_doc_info = pd.read_csv(reference_path+'file_mapping.csv', index_col=False)
    doc_info_dict = json.loads(df_doc_info.to_json(orient="records", index=False))
    try:
        (url, last_modified) = next((doc_info['url'], doc_info['modified_time']) 
                                    for doc_info in doc_info_dict if file == doc_info['name'])
    except StopIteration:
        (url, last_modified) = (None, None)
    return [url, last_modified]


def create_BM25_retriever(folder_path, retriever_name):
    all_docs = []

    for path, subdirs, files in os.walk(folder_path):
        for name in files:
            file = os.path.join(path, name)
            folder_name = os.path.basename(path)
            file_url_source = obtain_document_info(name)[0]
            file_last_modified = obtain_document_info(name)[1]
                
            if file.endswith(('.pdf','.docx','.pptx')):
                loader = UnstructuredFileLoader(file_path=file,mode='paged',post_processors=[clean_extra_whitespace])
                documents = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(separators=['\n'], chunk_size=2000, chunk_overlap=100)
                docs = text_splitter.split_documents(documents)
                
                for i, doc in enumerate(docs):
                    #Add file details to metadata of parent documents
                    doc.metadata['source'] = file_url_source
                    doc.metadata['last_modified'] = file_last_modified
                    doc.metadata['file_chunk_index'] = i
                    doc.metadata['type'] = folder_name
                    try:
                        page = doc.metadata['page_number']
                    except:
                        doc.metadata['page_number'] = 1
                    
                    all_docs.extend([doc])
            
            elif file.endswith(('.xlsx', '.xls', '.csv', '.xlsm')):
                documents = json.loads(cf.retrieve_multi_excel_text([file]))
                file_url_source = obtain_document_info(name)[0]
                file_last_modified = obtain_document_info(name)[1]
                for sheet_name, docs in documents.items():   
                    for i, doc in enumerate(docs):
                        doc = Document(page_content=json.dumps(doc), 
                                       metadata= {'filename':name
                                                  ,'page_number': sheet_name
                                                  ,'source': file_url_source
                                                  ,'last_modified': file_last_modified
                                                  ,'file_chunk_index': i
                                                  ,'type': folder_name
                                                  })
                        all_docs.extend([doc])
    
    output_log_path = user_path + 'output_logs/'        
    processed_files_w_url = []
    for doc in all_docs:
        processed_files_w_url.append([doc.metadata['filename'], doc.metadata['page_number'], doc.metadata['source'], doc.metadata['last_modified']])
    df_final_documents = pd.DataFrame(processed_files_w_url)
    df_final_documents.to_csv(output_log_path+f'{retriever_name}_processed_files.csv', index=False)

    retriever = BM25Retriever.from_documents(all_docs)
    save_object(retriever, f"{retriever_name}")
    return f"Retriever <<{retriever_name}>> created successfully"


policies_path = user_path + 'policies/'
#create retrievers
create_BM25_retriever(policies_path+'gl', "gl_bge_retriever-BM25-euclidean")
create_BM25_retriever(policies_path+'finance',"finance_bge_retriever-BM25-euclidean")
create_BM25_retriever(policies_path+'procurement',"procurement_bge_retriever-BM25-euclidean")
create_BM25_retriever(policies_path+'it',"it_bge_retriever-BM25-euclidean")