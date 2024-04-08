import os
import re
import pandas as pd
import PyPDF2
import anthropic
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders.excel import UnstructuredExcelLoader
from langchain.schema.document import Document
from unstructured.cleaners.core import clean_extra_whitespace
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chat_models import ChatAnthropic
import time

# MACOS directory
#user_path = '/Users/tammyauyong/Documents/Projects/bot-alex/index-creation/'
#policies_path = '/Users/tammyauyong/Documents/Projects/bot-alex/index-creation/Policies/'
user_path = 'C:/Users/auyon/Documents/bot-alex/index-creation/index-creation/'
policies_path = 'C:/Users/auyon/Documents/bot-alex/index-creation/policies/'
reference_path = user_path + 'reference/'

with open(reference_path+'anthropic_api_key.txt', 'r') as f:
    anthropic_api_key = f.read()
os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

def extract_summary(summary_text):
    pattern = r'<summary>(.*?)</summary>'
    match = re.search(pattern, summary_text, re.DOTALL)

    if match:
        extracted_text = match.group(1).strip()
    else:
        extracted_text = ''
    return extracted_text


def document_summary_chain(docs):
    # Define LLM chain
    llm = ChatAnthropic(model='claude-2.1', temperature=0.0, max_tokens_to_sample=2048, streaming=True)
    
    # Map
    map_template = """The following is a paragraph from a company policy document:
    {docs}
    Based on this list of docs, please summarise the information of the policy document below in a concised manner within 200 words. 
    Answer:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce
    reduce_template = """The following is set of summaries:
    {docs}
    Take these and distill it into a final, consolidated summary in 3 sentences of the objective of the policy. 
    Answer inside <summary></summary> XML tags.
    Answer"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    # Run chain
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
    
    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=12000,
    )

    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
        )
    
    for attempt in range(1,4):
        try:
            document_summary_text = map_reduce_chain.run(docs)
        except:
            print(f"Error: Retrying in 20 seconds.")
            time.sleep(20)

            if attempt == max_retries:
                print(f"Max retries reached. Unable to make a successful API call.")
                return None

    return document_summary_text

def create_document_summary(folder_path):
    df_document_summaries = pd.read_excel(reference_path+'document_summaries.xlsx', index_col=None)

    document_wo_summaries = []
    
    for path, subdirs, files in os.walk(folder_path):
        for name in files:
            file = os.path.join(path, name)
            if name not in df_document_summaries.filename.values:
                if file.endswith(('.pdf','.docx','.pptx')):
                    loader = UnstructuredFileLoader(file_path=file,mode='paged',post_processors=[clean_extra_whitespace])
                    documents = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(separators=['\n'], chunk_size=2048, chunk_overlap=256)
                    docs = text_splitter.split_documents(documents)
                    document_summary = document_summary_chain(docs)
                    
                    try:
                        extracted_summary = extract_summary(document_summary)
                        df_new_document_summaries = pd.DataFrame(data={'filename': [name], 'summary': [extracted_summary]})
                        df_document_summaries = pd.concat([df_document_summaries,df_new_document_summaries])
                        df_document_summaries.to_excel(reference_path+'document_summaries.xlsx', index=False)
                    except:
                        document_wo_summaries.append(name)
                        df_document_wo_summaries = pd.DataFrame(document_wo_summaries, columns=['filename'])
                        df_document_wo_summaries.to_excel(reference_path+'document_wo_summaries.xlsx', index=False)

                else:
                    document_wo_summaries.append(name)
                    df_document_wo_summaries = pd.DataFrame(document_wo_summaries, columns=['filename'])
                    df_document_wo_summaries.to_excel(reference_path+'document_wo_summaries.xlsx', index=False)

    return f"Document Summaries updated successfully"
            

#create index embeddings
create_document_summary(policies_path+'gl')
create_document_summary(policies_path+'finance')
create_document_summary(policies_path+'procurement')