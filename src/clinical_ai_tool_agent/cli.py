"""Interactive REPL: ask questions, see the ReAct trace.

Usage:
    python -m clinical_ai_tool_agent.cli
"""

from __future__ import annotations


def main() -> None:
    from clinical_ai_tool_agent.agent import build_default_llm, run_pipeline_react
    from clinical_ai_tool_agent.knowledge import build_retriever
    from clinical_ai_tool_agent.tools import build_tools

    print("clinical-ai-tool-agent  |  EDUCATIONAL USE ONLY  |  Ctrl-C to quit")
    print("Loading retriever and tools (first call may download embeddings)...")

    llm = build_default_llm()
    retriever = build_retriever()
    tools = build_tools()

    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            print("bye.")
            break

        trace = run_pipeline_react(
            prompt, llm=llm, retriever=retriever, tools=tools
        )

        if trace.tool_calls:
            print("\nTool calls:")
            for name, snippet in trace.tool_calls:
                print(f"  - {name}: {snippet}")

        print(f"\n--- Final Answer ({trace.steps} step(s), {trace.latency_s}s) ---")
        print(trace.final_answer or "(no final answer; max steps reached)")


if __name__ == "__main__":
    main()
