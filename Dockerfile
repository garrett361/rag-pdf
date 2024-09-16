FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev ffmpeg libsm6 libxext6 libmagic-dev libpoppler-dev poppler-utils

COPY rag/requirements.txt ./

RUN pip install -r requirements.txt

COPY rag/ .

CMD ["python", "parsing.py"]
