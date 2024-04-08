FROM python:3.11.1

WORKDIR /ah/retriever-creation

COPY requirements.txt .

RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx

CMD ["python", 'retriever_creation.py']