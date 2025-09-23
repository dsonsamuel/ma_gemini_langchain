FROM python:3.10.6-alpine
WORKDIR /python-pipeline-samples
ADD . /python-pipeline-samples
RUN pip install -r requirements.txt
CMD [" uvicorn main:app --reload --port 8002 &"]

