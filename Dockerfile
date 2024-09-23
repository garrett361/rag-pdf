FROM garrettgoon/rag-pdf-base

WORKDIR /app

COPY pyproject.toml ./

COPY . .
RUN pip install -e .
