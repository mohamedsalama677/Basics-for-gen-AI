"""CLI entry point.

    python src/query.py "What is self-attention?"
"""

import argparse

from graph import build_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the RAG pipeline a question.")
    parser.add_argument("question", help="The question to ask.")
    args = parser.parse_args()

    graph = build_graph()
    result = graph.invoke({"question": args.question})

    scored = result.get("scored_chunks") or []
    if scored:
        print("\n=== Retrieved Chunks ===")
        for doc, score in scored:
            src = doc.metadata.get("source", "?")
            cid = doc.metadata.get("chunk_id", "?")
            preview = doc.page_content[:200].replace("\n", " ")
            print(f"  [{src}#chunk_{cid}] score={score:.3f}")
            print(f"    {preview}...")
            print()

    print("=== Answer ===")
    print(result["answer"])

    citations = result.get("citations") or []
    if citations:
        print("\n=== Citations ===")
        for c in citations:
            print(f"- {c['source']}#chunk_{c['chunk_id']}")
            print(f"    {c['snippet']}...")
    else:
        print("\n(No citations — out-of-scope or refusal branch.)")

    print(f"\n(best retrieval score: {result.get('best_score'):.3f}, "
          f"has_context: {result.get('has_context')})")


if __name__ == "__main__":
    main()
