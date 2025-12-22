# RAG (Retrieval-augmented generation) ChatBot

[![CI](https://github.com/ms1104n/rag-chatbot/workflows/CI/badge.svg)](https://github.com/ms1104n/rag-chatbot/actions/workflows/ci.yaml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Check out the todo list to see the next steps and improvements implemented in this project [here](notes/todo.md).

> [IMPORTANT]
> Disclaimer:
> The code has been tested on:
> * Ubuntu 22.04.2 LTS running on a Lenovo Legion 5 Pro with twenty 12th Gen Intel Core i7-12700H and an NVIDIA GeForce RTX 3060.
> * MacOS Sonoma 14.3.1 running on a MacBook Pro M1 (2020).
>
> If you are using another Operating System or different hardware, and you can't load the models, please take a look at the official Llama Cpp Python's GitHub issues.

> [WARNING]
> - llama_cpp_python doesn't use GPU on M1 if you are running an x86 version of Python.
> - It's important to note that the large language model sometimes generates hallucinations or false information.

> [NOTE]
> To decide which hardware to use/buy to host your local LLMs we recommend you to read these benchmarks:
> - Performance of llama.cpp on Nvidia CUDA
> - Performance of llama.cpp on Apple Silicon M-series
>
> Decision model:
> - Memory capacity is the main limit. Check if your model fits in memory (with quantization).
> - Memory bandwidth mostly determines speed (tokens/sec). Check if the bandwidth gives you acceptable speed.
> - If not, upgrade hardware or optimize the model.
>
> For instance, it seems better to buy a second-hand or refurbished Mac Studio M2 Max with at least 64GB RAM, since it has 400Gbps of memory bandwidth compared to the M4 Pro, which has just 273Gbps.

## Table of contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
    - [Install Poetry](#install-poetry)
- [Bootstrap Environment](#bootstrap-environment)
    - [How to use the make file](#how-to-use-the-make-file)
    - [Environment](#environment)
- [Using the Open-Source LLMs/Embedding Models Locally](#using-the-open-source-llmsembedding-models-locally)
    - [Supported LLMs Models](#supported-llms-models)
    - [Supported Embedding Models](#supported-embedding-models)
- [Supported Response Synthesis strategies](#supported-response-synthesis-strategies)
- [Build the memory index](#build-the-memory-index)
- [Run the Chatbot](#run-the-chatbot)
- [References](#references)

## Introduction

This project combines the power of llama.cpp and Chroma to build:

* a Conversation-aware Chatbot (ChatGPT like experience).
* a RAG (Retrieval-augmented generation) ChatBot.

The RAG Chatbot works by taking a collection of Markdown files as input and, when asked a question, provides the corresponding answer based on the context provided by those files.

![rag-chatbot-architecture-1.png](images/rag-chatbot-architecture-1.png)

> [NOTE]
> We decided to refactor the RecursiveCharacterTextSplitter class from LangChain to effectively chunk Markdown files without adding LangChain as a dependency.

The Memory Builder component of the project loads Markdown pages from the docs folder. It then divides these pages into smaller sections, calculates the embeddings (a numerical representation) of these sections with the Semantic Search models from Sentence Transformers, and saves them in an embedding database called Chroma for later use.

When a user asks a question, the RAG ChatBot retrieves the most relevant sections from the Embedding database. Since the original question can't be always optimal to retrieve for the LLM, we first prompt an LLM to rewrite the question, then conduct retrieval-augmented reading. The most relevant sections are then used as context to generate the final answer using a local language model (LLM). Additionally, the chatbot is designed to remember previous interactions. It saves the chat history and considers the relevant context from previous conversations to provide more accurate answers.

To deal with context overflows, we implemented two approaches:

* Create And Refine the Context: synthesize responses sequentially through all retrieved contents.
    * ![create-and-refine-the-context.png](images/create-and-refine-the-context.png)
* Hierarchical Summarization of Context: generate an answer for each relevant section independently, and then hierarchically combine the answers.
    * ![hierarchical-summarization.png](images/hierarchical-summarization.png)

The Memory Builder builds the vector database in an incremental way, which means that when a document changes, we only update the corresponding chunks in the vector store instead of rebuilding the whole index.

This is achieved through:
- Document-level metadata tracking: every chunk gets tagged with a source doc ID + version hash. When a doc changes, we regenerate chunks for that doc only, delete the old ones by metadata filter, and insert new ones.
- Incremental ingestion pipeline: the pipeline diffs source docs against what is already indexed (using those version hashes). Only changed/new docs get processed.
- Handling deletions: we keep a separate mapping table (doc_id -> chunk_ids) in a SQLite db so we can precisely target what to remove without scanning the whole store.

> [IMPORTANT]
> One thing to watch out for - if you ever swap embedding models, you must rebuild it from scratch since the vector spaces won't be compatible. Plan for that early.

## Prerequisites

* Python 3.12+
* GPU supporting CUDA 12.4+
* Poetry 2.3.0

For the UI:
* Node 22.12+
* Yarn 1.22+

### Install Poetry

Install Poetry with pipx by following the official documentation.

You must use the current adopted version of Poetry defined in the version/poetry file.

If you have poetry already installed and is not the right version, you can downgrade (or upgrade) poetry through:

```
poetry self update <version>
```

or with pipx:

```
pipx install poetry==<version> --force
```

## Bootstrap Environment

To easily install the dependencies and start the services we created a make file.

### How to use the make file

> [IMPORTANT]
> Run Setup as your init command (or after Clean).

* Check: `make check`
    * Use it to check that `which pip3` and `which python3` points to the right path.
* Setup:
    * Setup with NVIDIA CUDA acceleration: `make setup_cuda`
        * Creates an environment and installs all dependencies with NVIDIA CUDA acceleration.
    * Setup with Metal GPU acceleration: `make setup_metal`
        * Creates an environment and installs all dependencies with Metal GPU acceleration for macOS system only.
* Start: `make start`
    * Start both the backend and frontend ensuring that the backend is running and ready before launching the frontend.
* Update: `make update`
    * Update an environment and installs all updated dependencies.
* Tidy up the code: `make tidy`
    * Run Ruff check and format.
* Clean: `make clean`
    * Removes the environment and all cached files.
* Test: `make test`
    * Runs all tests using pytest.

### Environment

Copy .env.example -> .env and fill it in.

Copy /frontend/.env.example -> .env and fill it in.

## Using the Open-Source LLMs/Embedding Models Locally

We utilize the open-source library llama-cpp-python, a binding for llama-cpp, allowing us to utilize it within a Python environment. llama-cpp serves as a C++ backend designed to work efficiently with transformer-based models. Running the LLMs architecture on a local PC is possible through this library, enabling us to run them either on a CPU or GPU. Additionally, we use Quantization and 4-bit precision to reduce the number of bits required to represent the numbers. The quantized models are stored in GGUF format.

### Supported LLMs Models

| Model | Supported | Model Size | Max Context Window | Notes and link to the model card |
|-------|-----------|------------|--------------------|----------------------------------|
| qwen-3.5:0.8b Qwen 3.5 0.8B | Yes | 0.8B | 256k | Tiny and fast multimodal, great for edge device |
| qwen-3.5:2b Qwen 3.5 2B | Yes | 2B | 256k | Multimodal for lightweight agents (small tool calls) |
| qwen-3.5:4b Qwen 3.5 4B | Yes | 4B | 256k | Does not drift from tasks as bad as 2B |
| qwen-3.5:9b Qwen 3.5 9B | Yes | 9B | 256k | Recommended model. Can handle complex tasks |
| qwen-2.5:3b - Qwen2.5 Instruct | Yes | 3B | 128k | Standard instruct model |
| qwen-2.5:3b-math-reasoning | Yes | 3B | 128k | Specialized for math reasoning |
| llama-3.2:1b Meta Llama 3.2 | Yes | 1B | 128k | Optimized for mobile or edge devices |
| llama-3.2 Meta Llama 3.2 | Yes | 3B | 128k | Optimized for mobile or edge devices |
| llama-3.1 Meta Llama 3.1 | Yes | 8B | 128k | Recommended model for general tasks |
| deep-seek-r1:7b - DeepSeek R1 | Yes | 7B | 128k | Experimental reasoning model |
| openchat-3.6 - OpenChat 3.6 | Yes | 8B | 8192 | High performance open model |
| openchat-3.5 - OpenChat 3.5 | Yes | 7B | 8192 | Previous stable version |
| starling Starling Beta | Yes | 7B | 8192 | Preferred for more verbosity |
| phi-3.5 Phi-3.5 Mini Instruct | Yes | 3.8B | 128k | Microsoft lightweight model |
| stablelm-zephyr StableLM | Yes | 3B | 4096 | Optimized for chat |

### Supported Embedding Models

For semantic search, we support all embedding models from Sentence Transformers. We recommend using the jina-embeddings-v5-text models, which are small with SOTA performance for multilingual retrieval tasks.

| Embedding Model | Supported | Model Size | Max Tokens | Retrieval score (MTEB) | Notes |
|-----------------|-----------|------------|------------|------------------------|-------|
| all-MiniLM-L6-v2 | Yes | 0.023B | 512 | 33.30 | Fast and lightweight |
| all-MiniLM-L12-v2 | Yes | 0.033B | 256 | 33.37 | Balanced performance |
| all-mpnet-base-v2 | Yes | 0.109B | 384 | 33.80 | High quality embeddings |
| jina-embeddings-v5-small | Yes | 0.596B | 32k | 64.88 | Recommended model |
| jina-embeddings-v5-nano | Yes | 0.212B | 8k | 63.26 | Efficient multilingual |

## Supported Response Synthesis strategies

| Response Synthesis strategy | Supported | Notes |
|-----------------------------|-----------|-------|
| create-and-refine | Yes | Sequential synthesis |
| tree-summarization | Yes | Recommended - Hierarchical combination |

## Build the memory index

You can download Markdown pages and place them under the docs folder. Build the memory index by running:

```shell
make migrate_db
python chatbot/memory_builder.py --model-name jinaai/jina-embeddings-v5-text-small-retrieval --chunk-size 1000 --chunk-overlap 50
```

## Run the Chatbot

The Chatbot has a UI built with Vite, React and TypeScript, and a backend built with FastAPI that serves the LLMs through llama-cpp-python.

To install the UI dependencies, run:

```shell
cd frontend
yarn

# Create .env file
echo "VITE_API_URL=http://localhost:8000" > .env
```

To start the backend:

```shell
cd backend && PYTHONPATH=.:../chatbot uvicorn main:app --reload
```

To start the frontend (in a new terminal):
```shell
cd frontend && yarn dev
```

Alternatively, start both using:

```shell
make start
```

The application will be available at http://localhost:5173, with the backend API at http://localhost:8000.

![conversation-aware-chatbot.gif](images/conversation-aware-chatbot.gif)

You can enable the RAG Mode feature in the UI to ask questions based on the context provided by the Markdown files you loaded and indexed.

![rag_chatbot_example.gif](images/rag_chatbot_example.gif)

## References

* Large Language Models (LLMs):
    * LLMs as a repository of vector programs
    * GPT in 60 Lines of NumPy
    * Calculating GPU memory for serving LLMs
    * Introduction to Weight Quantization
    * Uncensor any LLM with abliteration
    * Understanding Multimodal LLMs
    * Direct preference optimization (DPO): Complete overview
* LLM Frameworks:
    * llama.cpp and llama-cpp-python
    * Deepval - A framework for evaluating LLMs
    * Structured Outputs (Outlines)
* LLM Datasets:
    * High-quality datasets (mlabonne)
* Agents:
    * Building effective agents (Anthropic)
    * PydanticAI and Atomic Agents
    * agno - lightweight agent library
* Vector Databases:
    * Indexing algorithms: HNSW and IVF
    * Nearest Neighbor Indexes for Similarity Search
    * Chroma and Qdrant
* Retrieval Augmented Generation (RAG):
    * Building A Generative AI Platform
    * Rewrite-Retrieve-Read
    * Rerank and Response Synthesis
    * Conversational awareness

---

## Maintainer

**Sai Nikhil Mattapalli**
AI/ML Engineer

Sai is an AI/ML Engineer with over 5 years of experience building and deploying scalable Machine Learning and Generative AI solutions. He specializes in end-to-end ML pipelines, RAG architectures, and LLM integration.

* Email: ms1104n@gmail.com
* Role: AI/ML Engineer
* Expertise: Python, PyTorch, TensorFlow, LangChain, FastAPI, and Vector Databases.