.PHONY: install
install:
	pip install -e .

.PHONY: fmt
fmt:
	ruff format rag

.PHONY: check
check:
	ruff check rag

.PHONY: clean
clean:
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf *.egg-info/
	rm -rf pip-wheel-metadata/
	rm -rf build/
	find . \( -name __pycache__ -o -name \*.pyc \) -delete
	rm -rf private/test/*

.PHONY: test-parse
test-parse:
	python -m rag.parse --input private/RFQ_Commercial/NZT --output private/test/parsed --chunking_strategy "by_title"

.PHONY: test-embed
test-embed:
	python -m rag.embed --data-path private/test/parsed --path-to-db private/test/embedded

.PHONY: test-query
test-query:
	python -m rag.query "What is the output size/quantity of the plant?" --path-to-db private/test/embedded --model-name meta-llama/Meta-Llama-3.1-8B-Instruct --top-k-retriever 5

.PHONY: test
test:
	$(MAKE) clean
	$(MAKE) test-parse
	$(MAKE) test-embed
	$(MAKE) test-query
