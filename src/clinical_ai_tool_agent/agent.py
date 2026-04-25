"""ReAct loop tying together: retriever -> LLM -> parser -> tool -> ...

`run_pipeline_react` runs up to MAX_STEPS rounds. Each round retrieves
context, sends a fresh message stack to the LLM, parses the reply, and
either (a) returns a Final Answer or (b) executes a tool and feeds the
result back to the LLM on the next round.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from clinical_ai_tool_agent.parser import parse_action_or_final
from clinical_ai_tool_agent.prompts import (
    REACT_FORMAT_INSTRUCTIONS,
    SYSTEM_INSTRUCTIONS,
)


MAX_STEPS = 5


@dataclass
class AgentTrace:
    final_answer: str | None = None
    steps: int = 0
    tool_calls: list[tuple[str, str]] = field(default_factory=list)
    latency_s: float = 0.0
    raw_outputs: list[str] = field(default_factory=list)


def _combine_docs(docs, k: int = 3) -> str:
    return "\n".join(d.page_content for d in docs[:k]) if docs else "(none)"


def _build_chat_prompt(system: str, format_block: str, context: str, question: str):
    from langchain_core.prompts import ChatPromptTemplate

    template = ChatPromptTemplate.from_messages([
        ("system", "{system}\n\n{format_block}"),
        ("system", "RETRIEVED_CONTEXT:\n{context}"),
        ("human", "{question}"),
    ])
    return template.format_messages(
        system=system,
        format_block=format_block,
        context=context,
        question=question,
    )


def _call_tool(tools, name: str, args: dict | None) -> str:
    target = next((t for t in tools if t.name == name), None)
    if target is None:
        return f"ERROR: tool '{name}' not found."
    try:
        return target.invoke({} if args is None else args)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


def run_pipeline_react(
    user_prompt: str,
    *,
    llm,
    retriever,
    tools,
    max_steps: int = MAX_STEPS,
) -> AgentTrace:
    """Run the ReAct loop until a Final Answer is produced or max_steps hit."""
    from clinical_ai_tool_agent.tools import tool_summaries

    trace = AgentTrace()
    t_start = time.time()
    question = user_prompt
    format_block = REACT_FORMAT_INSTRUCTIONS.format(tool_list=tool_summaries(tools))

    for step in range(1, max_steps + 1):
        trace.steps = step
        docs = retriever.invoke(question)
        context = _combine_docs(docs, k=3)

        messages = _build_chat_prompt(
            SYSTEM_INSTRUCTIONS, format_block, context, question
        )
        ai = llm.invoke(messages)
        ai_text = (
            ai.content if hasattr(ai, "content") and isinstance(ai.content, str)
            else str(ai)
        )
        trace.raw_outputs.append(ai_text)

        tool_name, tool_args, final = parse_action_or_final(ai_text)

        if final is not None:
            trace.final_answer = final
            break

        if tool_name:
            result = _call_tool(tools, tool_name, tool_args)
            trace.tool_calls.append((tool_name, str(result)[:200]))
            question = (
                f"{user_prompt}\n\nTool Result ({tool_name}): {result}"
            )
            continue

        # Neither tool nor final — nudge the model to commit.
        question = (
            f"{user_prompt}\n(Please provide either a valid tool call or a Final Answer.)"
        )

    trace.latency_s = round(time.time() - t_start, 3)
    return trace


def build_default_llm(model: str = "tinyllama", host: str = "http://localhost:11434"):
    from langchain_community.chat_models import ChatOllama

    return ChatOllama(
        base_url=host,
        model=model,
        temperature=0.2,
        max_tokens=512,
        top_p=0.9,
        top_k=40,
    )
