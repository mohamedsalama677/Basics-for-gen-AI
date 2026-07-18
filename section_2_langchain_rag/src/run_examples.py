"""Runs the three assessment example questions through the graph and writes
the actual outputs to examples/qa_examples.md.

    python src/run_examples.py
"""

from pathlib import Path

from config import SECTION_ROOT
from graph import build_graph

EXAMPLES = [
    "What is self-attention in the transformer architecture?",
    "When would you choose RAG over fine-tuning?",
    "What is the capital of France?",
]

OUTPUT_PATH = SECTION_ROOT / "examples" / "qa_examples.md"


def main() -> None:
    graph = build_graph()

    lines: list[str] = [
        "# RAG Pipeline — Example Q&A",
        "",
        "Actual outputs from `run_examples.py`. Questions 1 and 2 are in-KB; "
        "question 3 is out-of-KB and must hit the refusal branch.",
        "",
    ]

    for i, question in enumerate(EXAMPLES, start=1):
        print(f"[{i}/{len(EXAMPLES)}] {question}")
        result = graph.invoke({"question": question})

        lines.append(f"## Question {i}")
        lines.append("")
        lines.append(f"**Q:** {question}")
        lines.append("")
        lines.append(f"**A:** {result['answer']}")
        lines.append("")

        citations = result.get("citations") or []
        if citations:
            lines.append("**Citations:**")
            lines.append("")
            for c in citations:
                lines.append(f"- `{c['source']}#chunk_{c['chunk_id']}` — {c['snippet']}...")
            lines.append("")

        lines.append(
            f"_(best retrieval score: {result.get('best_score'):.3f}, "
            f"has_context: {result.get('has_context')})_"
        )
        lines.append("")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
