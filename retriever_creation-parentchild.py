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
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings, SentenceTransformerEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders.excel import UnstructuredExcelLoader
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.cleaners.core import clean_extra_whitespace
#from APIEmbeddings import *


#user_path = '/Users/tammyauyong/Documents/Projects/bot-alex/index-creation/'
#policies_path = '/Users/tammyauyong/Documents/Projects/bot-alex/index-creation/Policies/'
user_path = 'C:/Users/auyon/Documents/bot-alex/index-creation/'
reference_path = user_path + 'index-creation/reference/'

def save_object(obj, filename):
    retriever_path = user_path + 'index-creation/retriever/'
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

def create_multi_vector_retriever(folder_path, retriever_name):
    #API_URL = "http://dpo.asuscomm.com:8088/predict"
    #embedding = APIEmbeddings(API_URL)
    embedding = HuggingFaceBgeEmbeddings(model_name="baai/bge-large-en-v1.5", encode_kwargs = {'normalize_embeddings': True})
    all_parent_docs = []
    all_doc_ids = []
    all_child_docs = [] 

    # The storage layer for the parent documents
    store = InMemoryStore()
    id_key = "doc_id"
    for path, subdirs, files in os.walk(folder_path):
        for name in files:
            file = os.path.join(path, name)
            folder_name = os.path.basename(path)
            file_url_source = obtain_document_info(name)[0]
            file_last_modified = obtain_document_info(name)[1]
                
            if file.endswith(('.pdf','.docx','.pptx')):
                loader = UnstructuredFileLoader(file_path=file,mode='paged',post_processors=[clean_extra_whitespace])
                documents = loader.load()
                
                parent_splitter = RecursiveCharacterTextSplitter(separators=['\n'], chunk_size=2000)
                parent_docs = parent_splitter.split_documents(documents)
                doc_ids = [str(uuid.uuid4()) for _ in parent_docs]
                
                all_parent_docs.extend(parent_docs)
                all_doc_ids.extend(doc_ids)
                
                
                for i, parent_doc in enumerate(parent_docs):
                    #Add file details to metadata of parent documents
                    parent_doc.metadata['source'] = file_url_source
                    parent_doc.metadata['last_modified'] = file_last_modified
                    parent_doc.metadata['file_chunk_index'] = i
                    parent_doc.metadata['type'] = folder_name
                    try:
                        page = parent_doc.metadata['page_number']
                    except:
                        parent_doc.metadata['page_number'] = 1

                    #create child documents from parent document
                    _id = doc_ids[i]
                    child_docs = []
                    child_splitter = RecursiveCharacterTextSplitter(separators=['\n'], chunk_size=400)
                    _child_docs = child_splitter.split_documents([parent_doc])
                    for _child_doc in _child_docs:
                        _child_doc.metadata[id_key] = _id
                    
                    child_docs.extend(_child_docs)
                    all_child_docs.extend(_child_docs)
            
            elif file.endswith(('.xlsx', '.xls', '.csv', '.xlsm')):
                documents = json.loads(cf.retrieve_multi_excel_text([file]))
                file_url_source = obtain_document_info(name)[0]
                file_last_modified = obtain_document_info(name)[1]
                for sheet_name, parent_docs in documents.items():
                    child_docs = []
                    doc_ids = [str(uuid.uuid4()) for _ in parent_docs]
                    all_doc_ids.extend(doc_ids)
                    
                    for i, parent_doc in enumerate(parent_docs):
                        doc = Document(page_content=json.dumps(parent_doc), 
                                       metadata= {'filename':name
                                                  ,'page_number': sheet_name
                                                  ,'source': file_url_source
                                                  ,'last_modified': file_last_modified
                                                  ,'file_chunk_index': i
                                                  ,'type': folder_name
                                                  })
                        all_parent_docs.extend([doc])
                        
                        #create child documents from parent document - in this case, parent & child docs are the same
                        _id = doc_ids[i]
                        _child_doc = doc
                        _child_doc.metadata[id_key] = _id
                        child_docs.extend([_child_doc])
                    all_child_docs.extend(child_docs)
    
    output_log_path = user_path + 'index-creation/output_logs/'        
    processed_files_w_url = []
    for doc in all_parent_docs:
        processed_files_w_url.append([doc.metadata['filename'], doc.metadata['page_number'], doc.metadata['source'], doc.metadata['last_modified']])
    df_final_documents = pd.DataFrame(processed_files_w_url)
    df_final_documents.to_csv(output_log_path+f'{retriever_name}_processed_files.csv', index=False)

    #create MultiVectorRetriever
    vectorstore = FAISS.from_documents(documents=all_child_docs,embedding=embedding, distance_strategy='MAX_INNER_PRODUCT', normalize_L2=True)
    # Create a multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore
        ,docstore=store
        ,id_key=id_key
        )
    
    retriever.docstore.mset(list(zip(all_doc_ids, all_parent_docs)))
    save_object(retriever, f"{retriever_name}")
    return f"Retriever <<{retriever_name}>> created successfully"


policies_path = user_path + 'policies/'
#create retrievers
create_multi_vector_retriever(policies_path+'gl', "gl_bge_retriever-parentchild")
create_multi_vector_retriever(policies_path+'finance',"finance_bge_retriever-parentchild")
create_multi_vector_retriever(policies_path+'procurement',"procurement_bge_retriever-parentchild")