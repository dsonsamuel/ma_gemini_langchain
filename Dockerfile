FROM innersource-artefacts-docker.lloydsbanking.cloud/ingested/python:3.11-slim AS builder
WORKDIR /ma_gemini_langchain
ADD . /ma_gemini_langchain
#RUN pip install -r requirements.txt
RUN pip install -r requirements.txt
CMD ["cd", "src"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001" ]

