import argparse
import sys
import os

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ingest import run_ingest
from query  import run_query


def main():
    parser = argparse.ArgumentParser(
        description="Biomedical RAG pipeline — ingest abstracts, then query them."
    )
    parser.add_argument("--query",  default="", help="PubMed search query (optional)")
    parser.add_argument("--email",  default="", help="Your email for NCBI Entrez (required for PubMed)")
    parser.add_argument("--max",    default=10, type=int, help="Max PubMed results")
    parser.add_argument("--model",  default="mistral", help="Ollama model (default: mistral)")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip ingest and go straight to querying (use existing ChromaDB)")
    args = parser.parse_args()

    #ingest
    if not args.skip_ingest:
        print("=" * 60)
        print("STEP 1: INGESTING ABSTRACTS")
        print("=" * 60)
        run_ingest(
            pubmed_query=args.query,
            pubmed_email=args.email,
            max_pubmed=args.max,
        )
    else:
        print("[main] Skipping ingest — using existing ChromaDB.\n")

    #query loop
    print("=" * 60)
    print("STEP 2: QUERY MODE  (type 'quit' to exit)")
    print("=" * 60)

    while True:
        try:
            question = input("\nAsk a question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        answer = run_query(question, verbose=False)

        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(answer)


if __name__ == "__main__":
    main()