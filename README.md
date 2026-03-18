# Biomedical-RAG

A local Retrieval-Augmented Generation (RAG) pipeline for biomedical literature. This project enables the user to answer questions and the responses are grounded from relevant PubMed abstracts.

Pipeline: User question --> Embedding --> Retrieval --> Generation

Embedding: The question is encoded as a vector using a SentenceTransformer model to capture keywords and semantic meaning.

Retrieval: ChromaDB stores pre-computed embeddings for abstracts, and uses a Hierarchical Navigable Small Worlds (HNSW) index to find top-k abstracts whose vectors are most similar to the query vector. 

Generation: The retrieved abstracts are injected into a prompt template as contenxt, and the prompt is sent to a local LLM like Mistral via Ollama. 


## Setup

For Python dependencies
```
pip install -r requirements.txt
```
Install Ollama from https://ollama.com, then pull a local model. I pulled Mistral for my trials. 

```
ollama pull mistral
```

Check the Ollama server has started by running:
```
ollama serve
```

## Running the application

```
# Uses built-in fallback abstracts if you don't put a query. This was for development and testing. 
python main.py

# Uses live PubMed abstracts (requires your email for NCBI Entrez). Below is an example query
python main.py --query "LLMs in biomedical research" --email you@example.com

```

### Limitations 
This project currently has no chunking, so each abstract is stored as a single chunk. To retrieve longer documents, such as research papers and journals, splitting would be required. 
Additionally, a cross-encoder reranker can improve precision, as the currently retrieved results are ranked solely by vector similarity. There are numerous other possibilities to improve this system; this project was to practice building RAG systems.


