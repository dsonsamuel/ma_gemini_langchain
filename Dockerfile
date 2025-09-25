FROM python:3.10.6-alpine
RUN pwd
WORKDIR /ma_gemini_langchain
ADD . /ma_gemini_langchain
#RUN pip install -r requirements.txt
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--reload" ]

