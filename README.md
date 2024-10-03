<img
  src="https://raw.githubusercontent.com/hpe-design/logos/master/Requirements/color-logo.png"
  alt="HPE Logo"
  height="100"
/>

# Notes

Building on top of the rag-pdf demo below.

For ease of development, I am cannibalizing parts of the original code and putting them under
`rag/`.

I copied the `RFQ_Commercial` dir under `private/RFQ_Commercial` which is `.gitignore`-d. This is
only referenced in the `Makefile` which has some convenience commands. E.g. run `make test-parse` to
parse a test pdf in `RFQ_Commercial` and then `make test-embed` to embed the parsed results into a
vector db. Use `make test-query` to test querying a local LLM and `make test-query-hosted` to test
the hosted endpoint path. Append `QUERY=<your new query here>` to change the test query.

Run `make test` to run the entire `{parse,embed,query}` workflow from start to finish, and similar
for `make test-embed` to run the version with hosted model endpoints.

There are also multiple `requirements.txt`'s floating in various parts of the repo. The top-level
`rag/requirements.txt` is the one I have in my `venv` while developing, built from installing parts
of the other `requirements.txt` instances on top of each other.

Notes when running/developing locally on an M1 Mac.

For parsing:

- `brew install libmagic`
- `brew install poppler`
- `brew install tesseract`
- `pip install -U torch nltk`

For query:

- `pip install llama-index-llms-openllm==0.1.4`

# Experimental Features

- **Additional Post-Parsing Processing**:

  - Use the chat LLM to judge whether each chunk is informative by answering a yes/no question.
    Reject the chunk, if uninformative. See the `--filter-parsed-with-llm` flag for `rag/parse.py`

  - For each chunk, use the LLM to generate up to three question which can be answered by the
    chunk contents. These are added to the chunk as metadata, which affects the chunk embeddings and
    may improve retrieval. See the `--add-questions` flag for `rag/parse.py`

- **Querying**

  - Reranking with the chat LLM. New `LLama31Reranker` class ranks every chunk on a scale from 1 to 10, which is used to re-rank. The default when supplying a `--top-k-reranker` flag to `rag/query.py`.
  - The `LLama31Reranker` can either provide a fixed number of reranked results via the `--top-k-reranker` or return all results which surpass a minimum score threshold by setting `--min-score-reranker`. Mutually exclusive options.
    `--rerank-with-questions` flag. The questions are only added for re-ranking; the questions are
    not seen by the chat LLM at the generation step.
  - Can instead use a default Sentence Transformer to perform the reranking by supplying an additional `--st-reranker` flag. Not compatible with `--min-score-reranker`.
  - Optionally append the questions-answered metadata to the chunks before re-ranking with the
    `--rerank-with-questions` flag. The questions are only added for re-ranking; the questions are
    not seen by the chat LLM at the generation step.

# Other Notes

- See the many variants of commands in the `Makefile` for different workflows.
- The experimental additions aren't well incorporated into the gui, yet.
- This branch uses `Weaviate` in order to play with hybrid search, which mixes lexical and vector
  search together. The results are very iffy, though. Finding sometimes that a very relevant node is
  returned with a score of precisely 0.0, which seems like a bug. Finding best results with the (now
  default) `cutoff = None` for similarity search to avoid filtering out such results.

### Example Workflow

- Parse folder with docs and write chunks into `json` file via `python3 -m rag.parse --input <path-to-docs-dir>
--output <path-to-parse-output-dir>`
- Embed the parsed & chunked docs: `python -m rag.embed --data-path <path-to-parse-output-dir> --path-to-db <path-to-db>`
- Query using locally hosted LLM: `python -m rag.query "What is the name of the project?" --path-to-db <path-to-db> --model-name meta-llama/Meta-Llama-3.1-8B-Instruct --top-k-retriever 5`
- Query using endpoint-hosted embedding model and LLM: `python -m rag.query "What is the name of the project?" --path-to-db <path-to-db> --model-name meta-llama/Meta-Llama-3.1-70B-Instruct --top-k-retriever 5 --chat-model-endpoint http://llama-31-70b-jordan.models.mlds-kserve.us.rdlabs.hpecorp.net/v1/ --embedding_model_path http://embedding-tyler.models.mlds-kserve.us.rdlabs.hpecorp.net/v1`

## Docker

To create docker files for the entire workflow under `rag/`, run `build_base_image.sh` (which takes
care of requirements and can take a while), followed by `build_image.sh`. Both of these are located
at the root level. The final image will be tagged with the short git hash.

-------------- ORIGINAL README BELOW --------------

# RAG demo (Chat with HPE Press Release version)

<b>Author:</b> Tyler Britten, Andrew Mendez </br>
<b>Date:</b> 05/01/2024</br>
<b>Revision:</b> 0.1</br>

This demonstration was built to showcase Retrieval Augmented Generation (RAG) on HPE press release documents.
It shows how RAG can be used to assist customers with keeping up with recent HPE news articles.

To replicate this demo, you will need:

- A functioning Kubernetes cluster with load balancers for external facing services configured
  - cluster having shared mounted folder `/nvmefs1/tyler.britten`
- Pachyderm/HPE MLDM 2.9.2 installed on the cluster and fully functional
- At least 1x NVIDIA T4 80GB GPUs
- Determined.AI/HPE MLDE environment for finetuning models (not included in the base code here)
-

<b>NOTE:</b> You might be able to replicate this demo with other GPUs (for example L40s) as well, but you need to consider the memory footprint of other GPUs and adjust accordingly.

## Recorded demos of RAG demo

[ToDo]

## Implementation Overview

- Step 1: Connect to deployed MLDM application
- Step 2: Create MLDM project named `rag-demo-hpe`
- Step 3: Set new project as current context
- Step 4: Create repo `documents` to hold xml documents
- Step 5: Upload xml documents
- Step 6: Pipeline step to parse documents
- Step 7: Pipeline step to chunk documents
- Step 8: Pipeline step to embed documents using embedding model `bge-large-en-v1.5`
- Step 9: Pipeline step to deploy GUI application
- Step 10: Interact with GUI application
- Step 11: Add new documents to repo `documents` to improve RAG App
- Step 12: (Optional Step) Pipeline step to develop dataset for finetuning embeddings: qna pipeline
- Step 13: (Optional Step) Pipeline step to finetune embeddings
- Step 14: Delete pipelines

<b>RAG demo includes solution components from:</b>

- [Pachyderm / HPE MLDM](https://www.hpe.com/us/en/hpe-machine-learning-data-management-software.html)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit frontend](https://streamlit.io)

# Steps to run the demo

## Step 1: Connect to deployed MLDM application

`pachctl connect pachd-peer.pachyderm.svc.cluster.local:30653`

## Step 2: Create MLDM project named `rag-demo-hpe`

`pachctl create project rag-demo-hpe`

## Step 3: Set new project as current context

`pachctl config update context --project rag-demo-hpe`

### Pipeline will be available at the url:

`http://mldm-pachyderm.us.rdlabs.hpecorp.net/lineage/pdf-rag-andrew`

## Step 4: Create repo `documents` to hold xml documents

`pachctl create repo documents`

## Step 5: Upload xml documents

`pachctl put file documents@master: -f data/antonio-neri.xml`
`pachctl put file documents@master: -f data/aruba_wifi_7_press.xml`
`pachctl put file documents@master: -f data/e2e_ai_platform_press_release.xml`

## Step 6: Pipeline step to parse documents

This pipeline takes the raw xml documents and parses them into json format.

`pachctl create pipeline -f pipelines/parsing.pipeline.json`

## Step 7: Pipeline step to chunk documents

This pipeline takes the parsed documents and applies chunking to create chunked documents.

`pachctl create pipeline -f pipelines/chunking.pipeline.json`

## Step 8: Pipeline step to embed documents using embedding model `bge-large-en-v1.5`

This pipeline takes the chunked documents and creates vector embeddings using the vector embedding `bge-large-en-v1.5`

`pachctl create pipeline -f pipelines/embedding.pipeline.json`

## Step 9: Pipeline step to deploy GUI application

This pipeline will deploy a streamlit application for user to interact with the GUI

`pachctl create pipeline -f pipelines/gui.pipeline.json`

## Step 10: Interact with GUI application

Note: There is an issue with the Houston cluster where there are not enough IP addresses for service pipeline.

### run command to ssh into node:

`ssh andrew@mlds-mgmt.us.rdlabs.hpecorp.net -L 8080:localhost:8080`

### run command to port forward GUI to local computer:

`kubectl port-forward -n pachyderm svc/pdf-rag-andrew-gui-v1-user 8080:80`

### Open web browser and go to url `localhost:8080`

#### Ask in the UI:

`Who is Antonio Neri?`

also ask a question about a new announcement about HPE Aruba Networking:

`What did HPE Aruba Networking Introduce?`

#### Now ask, the app won't answer it correctly:

`Who is Neil MacDonald?`

## Step 11: Add new documents to repo `documents` to improve RAG App

We will show the key value proposition with a data driven pipeline, add more documents, and the RAG app will automatically be updated.

### In the terminal, add new document:

`pachctl put file documents@master: -f data/neil-macdonald.xml`

When pipeline is done, refresh webpage and ask:
`Who is Neil MacDonald?`

# Optional Steps

## Step 12: (Optional Step) Pipeline step to develop dataset for finetuning embeddings: qna pipeline

`pachctl create pipeline -f pipelines/qna.pipeline.json`

## Step 13: (Optional Step) Pipeline step to finetune embeddings

Note, what is hardcoded is the following in `finetune/experiment/const.yaml`:

Make sure you create a workspace named `Tyler` and project `doc_embeds` to use the same `finetune/experiment/const.yaml`

```
name: arctic-embed-fine-tune
workspace: Tyler
project: doc_embeds
```

Also, the bind_mounts are hardcoded. This assumes you are running on a cluster (i.e. the houston cluster) where you have a mounted shared folder called `/nvmefs1`

```
bind_mounts:
  - container_path: /nvmefs1/
    host_path: /nvmefs1/
    propagation: rprivate
    read_only: false
  - container_path: /determined_shared_fs
    host_path: /nvmefs1/determined/checkpoints
    propagation: rprivate
    read_only: false
```

### Step to trigger finetune pipeline:

`pachctl create pipeline -f pipelines/finetune.pipeline.json`

## Step 14: Delete pipelines

`pachctl delete pipeline gui`
`pachctl delete pipeline finetune-embedding`
`pachctl delete pipeline generate-qna`
`pachctl delete pipeline embed-docs`
`pachctl delete pipeline chunk-doc`
`pachctl delete pipeline parse-docs`
`pachctl delete repo documents`
