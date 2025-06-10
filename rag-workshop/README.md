# Workshop for RAG

## Requirements
- Docker
- An environment that can run an LLM or an LLM API key
- A python runtime (for the LLM client)

## Setup
- Install docker [Guide](https://docs.docker.com/engine/install/)
- Instal ollama [Guide](https://ollama.com/download)
- Get a python runtime [uv](https://docs.astral.sh/uv/getting-started/installation/)


## Guide

### Create running Vector database

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Prepare ollama, download embedding model, download LLM
```bash
ollama pull nomic-embed-text
ollama pull gemma3:4b

```

### Install python dependencies

```bash
uv init --python 3.12
uv sync
```

### Run example application

```bash
uv run main.py
```
