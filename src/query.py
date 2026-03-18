"""
query.py

Query the biomedical RAG pipeline.

  1. Embeds the question using the same model used during ingest
  2. Retrieves the top-k most similar abstracts from ChromaDB
  3. Passes those abstracts as context to a local LLM via Ollama
  4. Prints the answer

Run python src/ingest.py first to populate the vector store.
"""

import sys
import json
import urllib.request
import urllib.error
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH      = "./chroma_db"
COLLECTION_NAME  = "biomedical_abstracts"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
OLLAMA_URL       = "http://localhost:11434/api/generate"
OLLAMA_MODEL     = "mistral"   
TOP_K            = 3           # how many abstracts to retrieve per query


def embed_query(query: str) -> list:
    """
    Encode the user's question into a vector using the same embedding model
    used during ingest. This is critical — the query and documents must live
    in the same vector space for similarity search to work.
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model.encode([query])[0].tolist()


def retrieve(query_embedding: list, top_k: int = TOP_K) -> list:
    """
    Search ChromaDB for the top_k abstracts most similar to the query embedding.
    ChromaDB uses approximate nearest-neighbor search (HNSW index) over cosine distance.
    Returns a list of dicts: {id, title, text, distance}.
    """
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "id":       results["ids"][0][i],
            "title":    results["metadatas"][0][i].get("title", ""),
            "text":     results["documents"][0][i],
            "distance": results["distances"][0][i], # lower means more similar
        })
    return retrieved


def build_prompt(question: str, context_docs: list) -> str:
    """
    Assemble the prompt that will be sent to the LLM.
    The retrieved abstracts are injected as context so the model can
    ground its answer in the actual literature, rather than hallucinating.
    This is the core idea of RAG: retrieval-augmented generation.
    """
    context_blocks = []
    for i, doc in enumerate(context_docs, 1):
        context_blocks.append(f"[Abstract {i}] {doc['title']}\n{doc['text']}")
    context_str = "\n\n".join(context_blocks)

    prompt = f"""You are a biomedical research assistant. Answer the question below using ONLY the provided abstracts as your source. If the abstracts do not contain enough information to answer, say so clearly.

--- CONTEXT ---
{context_str}

--- QUESTION ---
{question}

--- ANSWER ---"""
    return prompt


def call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Send the prompt to a locally running Ollama instance via its REST API.
    Ollama exposes a simple HTTP endpoint — no paid API needed.
    The response is a stream of JSON lines; we collect all text chunks.
    """
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        full_response = []
        with urllib.request.urlopen(req, timeout=120) as response:
            for line in response:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                chunk = json.loads(line)
                #Each chunk here has a "response" field with a token or partial text
                full_response.append(chunk.get("response", ""))
                if chunk.get("done", False):
                    break
        return "".join(full_response).strip()

    except urllib.error.URLError:
        return (
            "[ERROR] Could not connect to Ollama. Make sure Ollama is running "
            f"(run: ollama serve) and the model '{model}' is pulled "
            f"(run: ollama pull {model})."
        )


def run_query(question: str, verbose: bool = False) -> str:
    """
    Full RAG pipeline: embed → retrieve → prompt → generate.
    Returns the LLM's answer as a string.
    """
    print(f"\n[Query] '{question}'")

    #embed the question
    print("[1/3] Embedding question...")
    q_embedding = embed_query(question)

    # retrieve relevant abstracts
    print(f"[2/3] Retrieving top {TOP_K} abstracts from ChromaDB...")
    docs = retrieve(q_embedding)

    if verbose:
        print("\n--- Retrieved Abstracts ---")
        for doc in docs:
            print(f"  [{doc['id']}] {doc['title']} (distance: {doc['distance']:.4f})")
        print()

    #prompt and call LLM
    print(f"[3/3] Querying Ollama ({OLLAMA_MODEL})... (this may take a moment)\n")
    prompt  = build_prompt(question, docs)
    answer  = call_ollama(prompt)

    return answer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query the biomedical RAG pipeline.")
    parser.add_argument("question",         help="Your biomedical question")
    parser.add_argument("--verbose", "-v",  action="store_true", help="Show retrieved abstracts")
    parser.add_argument("--model",          default=OLLAMA_MODEL, help="Ollama model name")
    args = parser.parse_args()

    OLLAMA_MODEL = args.model  #allow override from CLI
    answer = run_query(args.question, verbose=args.verbose)

    print("=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(answer)
    print()