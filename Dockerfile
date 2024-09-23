FROM garrettgoon/rag-pdf-base

WORKDIR /app

COPY pyproject.toml ./

COPY rag/ ./
RUN pip install -e .
