import PyPDF2
import streamlit as st
import pandas as pd
import json
import docx

####################################################################################  
def retrieve_multi_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()  
    return text

def retrieve_multi_excel_text(excel_files):
    text = {}
    for excel_file in excel_files:
        if excel_file.endswith(('.csv')):
            df_data = pd.read_csv(excel_file)
            df_data.dropna(how='all', inplace=True)
            df_data.drop_duplicates(inplace=True)
            json_text = json.loads(df_data.to_json(orient="records"))
            text["Sheet1"]= json_text
            sheet_name = ''
        else:
            df_all_sheets = pd.ExcelFile(excel_file)
            count=0
            for sheet_name in list(df_all_sheets.sheet_names):
                df_data = pd.read_excel(excel_file, sheet_name=sheet_name)
                df_data.dropna(how='all', inplace=True)
                df_data.drop_duplicates(inplace=True)
                json_text = json.loads(df_data.to_json(orient="records"))
                text[sheet_name]= json_text
                count+=1
    return json.dumps(text)

def retrieve_multi_docx_text(word_files):
    text = ""
    for word_file in word_files:
        doc = docx.Document(word_file)
        for para in doc.paragraphs:
            text+=f"\n{para.text}"
    return text

def jd_upload(upload_name):
    # create a upload file widget for a pdf
    pdf_file_01 = st.file_uploader(upload_name, type=["pdf"], accept_multiple_files=True)

    # if a pdf file is uploaded
    if pdf_file_01:
        return retrieve_multi_pdf_text(pdf_file_01) 
    else:
        return "" 

def cv_upload(upload_name):
    # create a upload file widget for a pdf
    pdf_file_02 = st.file_uploader(upload_name, type=["pdf"], accept_multiple_files=True)

   # if a pdf file is uploaded
    if pdf_file_02:
        return retrieve_multi_pdf_text(pdf_file_02)
    else:
        return ""  
#extract the message history dict into a string         
def extract_info(data):
    result = ""
    for item in data:
        if item["type"] == "human":
            result += 'Human: '+ item["data"]["content"] + " ; \n\n"
        elif item["type"] == "ai":
            result += 'AI: '+ item["data"]["content"] + "\n\n"
    return result

#extract the message dict of human input portion history into a list        
def extract_human_history(data):
    human_data = []
    for d in data:
        if d['type'] == 'human':
            human_data.append(d['data']['content'])
    return human_data

# Define a function that takes a list as input and returns the reversed list
def reverse_list(lst):
    return lst[::-1]

# Convert the list to a string, and add line break after each list item
def list_to_string(lst):
    return '\n\n'.join(str(i+1) + '. ' + str(item) for i, item in enumerate(lst))



####################################################################################  
