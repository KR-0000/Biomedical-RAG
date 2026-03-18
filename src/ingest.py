"""
ingest.py

Handles loading and embedding biomedical abstracts into the ChromaDB vector store.
Run this once before querying. It supports:
  - Fetching live abstracts from PubMed via Biopython (requires internet + email)
  - Falling back to a hardcoded set of sample abstracts
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict

CHROMA_PATH = "./chroma_db"          
COLLECTION_NAME = "biomedical_abstracts"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  

#Fallback abstracts (used when PubMed fetch fails or no query/email provided). 
FALLBACK_ABSTRACTS = [
    {
        "id": "fallback_1",
        "title": "Deep learning for medical image analysis",
        "text": (
            "Deep learning models, particularly convolutional neural networks (CNNs), "
            "have shown remarkable performance in medical image analysis tasks including "
            "radiology, pathology, and ophthalmology. Transfer learning from ImageNet "
            "allows models to generalize even with limited labeled medical data. "
            "Challenges include dataset shift, explainability, and regulatory approval."
        ),
    },
    {
        "id": "fallback_2",
        "title": "Large language models in clinical NLP",
        "text": (
            "Large language models (LLMs) such as GPT and BERT variants have been "
            "fine-tuned for clinical NLP tasks including named entity recognition, "
            "relation extraction, and clinical question answering. Domain-specific "
            "pre-training on biomedical corpora (e.g., PubMedBERT, BioBERT) "
            "consistently outperforms general-purpose models on biomedical benchmarks."
        ),
    },
    {
        "id": "fallback_3",
        "title": "Transformer architectures in genomics",
        "text": (
            "Transformer-based models have been adapted for genomic sequence modeling, "
            "learning long-range dependencies in DNA that are missed by CNNs. "
            "Enformer and Nucleotide Transformer demonstrate that attention mechanisms "
            "can predict gene expression and regulatory elements from raw sequence, "
            "opening new directions in functional genomics."
        ),
    },
    {
        "id": "fallback_4",
        "title": "Federated learning for privacy-preserving clinical AI",
        "text": (
            "Federated learning enables training machine learning models across multiple "
            "hospitals without sharing raw patient data. Each site trains locally and "
            "shares only model gradients. Studies show federated models approach "
            "centralized performance for tasks like chest X-ray classification and "
            "EHR-based mortality prediction, addressing privacy and compliance concerns."
        ),
    },
    {
        "id": "fallback_5",
        "title": "RAG pipelines for biomedical question answering",
        "text": (
            "Retrieval-Augmented Generation (RAG) combines dense retrieval with "
            "generative models to answer biomedical questions grounded in literature. "
            "A query is embedded and matched against a vector index of abstracts; "
            "retrieved passages are passed as context to an LLM. RAG reduces "
            "hallucination compared to closed-book LLMs and is well-suited for "
            "evidence-based clinical question answering."
        ),
    },
    {
        "id": "fallback_6",
        "title": "Graph neural networks for drug-target interaction",
        "text": (
            "Graph neural networks (GNNs) model molecules as graphs of atoms and bonds, "
            "learning molecular representations for predicting drug-target binding affinity. "
            "Message-passing frameworks aggregate neighborhood features, enabling "
            "end-to-end learning from chemical structure. GNNs outperform fingerprint-based "
            "baselines on standard drug-target interaction benchmarks."
        ),
    },
]


def fetch_pubmed_abstracts(query: str, max_results: int = 10, email: str = "") -> List[Dict]:
    """
    Attempt to fetch abstracts from PubMed using Biopython's Entrez interface.
    Returns a list of dicts with keys: id, title, text.
    Returns [] on any failure.
    """
    try:
        from Bio import Entrez
        if not email:
            print("[PubMed] No email provided; skipping live fetch.")
            return []
        Entrez.email = email
        print(f"[PubMed] Searching for: '{query}' (max {max_results} results)...")

        # search for IDs
        search_handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        search_results = Entrez.read(search_handle)
        search_handle.close()
        ids = search_results["IdList"]

        if not ids:
            print("[PubMed] No results found.")
            return []

        # fetch abstracts by ID
        fetch_handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="xml")
        records = Entrez.read(fetch_handle)
        fetch_handle.close()

        abstracts = []
        for record in records["PubmedArticle"]:
            try:
                pmid = str(record["MedlineCitation"]["PMID"])
                title = str(record["MedlineCitation"]["Article"]["ArticleTitle"])
                abstract_text = record["MedlineCitation"]["Article"].get("Abstract", {})
                text = str(abstract_text.get("AbstractText", ""))
                if text:
                    abstracts.append({"id": f"pubmed_{pmid}", "title": title, "text": text})
            except (KeyError, TypeError):
                continue  #skip malformed records

        print(f"[PubMed] Fetched {len(abstracts)} abstracts.")
        return abstracts

    except ImportError:
        print("[PubMed] Biopython not installed; falling back to local abstracts.")
        return []
    except Exception as e:
        print(f"[PubMed] Fetch failed ({e}); falling back to local abstracts.")
        return []


def embed_and_store(abstracts: List[Dict], chroma_path: str = CHROMA_PATH) -> None:
    """
    Embed each abstract using a sentence-transformers model and store in ChromaDB.
    ChromaDB is a local vector database — it stores embeddings on disk and supports
    fast similarity search (cosine distance by default).
    """
    print(f"\n[Embed] Loading embedding model '{EMBEDDING_MODEL}'...")
    #SentenceTransformer encodes text into dense vectors (embeddings).
    #all-MiniLM-L6-v2 produces 384-dimensional vectors and runs fast on CPU.
    model = SentenceTransformer(EMBEDDING_MODEL)

    #Build a ChromaDB client that saves to disk
    client = chromadb.PersistentClient(path=chroma_path)

    #Delete old collection if it exists so we start fresh each ingest
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        # ChromaDB will use cosine similarity when we query
        metadata={"hnsw:space": "cosine"},
    )

    texts = [a["text"] for a in abstracts]
    ids   = [a["id"]   for a in abstracts]
    metas = [{"title": a["title"]} for a in abstracts]

    print(f"[Embed] Embedding {len(texts)} abstracts...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # Store everything: the raw text, its embedding, and metadata (title)
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metas,
    )
    print(f"[Embed] Stored {len(texts)} abstracts in ChromaDB at '{chroma_path}'.")


def run_ingest(pubmed_query: str = "", pubmed_email: str = "", max_pubmed: int = 10):
    """
    Main ingest entry point. Tries PubMed first, falls back to hardcoded abstracts.
    """
    abstracts = []

    if pubmed_query and pubmed_email:
        abstracts = fetch_pubmed_abstracts(pubmed_query, max_results=max_pubmed, email=pubmed_email)

    if not abstracts:
        print("[Ingest] Using fallback abstracts.")
        abstracts = FALLBACK_ABSTRACTS

    embed_and_store(abstracts)
    print("\n[Ingest] Done. You can now run query.py to ask questions.\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest biomedical abstracts into ChromaDB.")
    parser.add_argument("--query",   default="",  help="PubMed search query (optional)")
    parser.add_argument("--email",   default="",  help="Your email for NCBI Entrez (required for PubMed)")
    parser.add_argument("--max",     default=10,  type=int, help="Max PubMed results (default 10)")
    args = parser.parse_args()

    run_ingest(pubmed_query=args.query, pubmed_email=args.email, max_pubmed=args.max)