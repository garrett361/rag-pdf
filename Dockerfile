FROM garrettgoon/rag-pdf-base

COPY . .

RUN pip install -r rag/requirements.txt --no-cache-dir

RUN make install

