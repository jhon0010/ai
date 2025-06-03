from __future__ import annotations
from typing import TypedDict, Literal, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

# ---------- 1. Shared, strongly-typed state ----------
class GraphState(TypedDict):
    messages: List[AIMessage | HumanMessage]

# ---------- 2. LLM setup (swap for Ollama, Together, etc.) ----------
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)

# ---------- 3. Agent node implementations ----------
def researcher(state: GraphState) -> Command[Literal["writer"]]:
    """Pull out the latest user question and return research bullets."""
    user_query = state["messages"][-1].content
    prompt = (
        "You are an expert researcher. Produce concise, referenced bullet-points\n"
        f"that answer: {user_query}"
    )
    resp = llm.invoke(prompt)
    return Command(
        goto="writer",
        update={"messages": state["messages"] + [AIMessage(name="Researcher", content=resp.content)]},
    )


def writer(state: GraphState) -> Command[END]:
    """Craft the final answer using all researcher bullets."""
    bullets = "\n".join(
        m.content for m in state["messages"] if isinstance(m, AIMessage) and m.name == "Researcher"
    )
    prompt = (
        "You are a senior technical writer. Using the research notes below, "
        "compose a clear, well-structured answer for the user.\n\n"
        f"=== NOTES ===\n{bullets}\n\n=== ANSWER ==="
    )
    resp = llm.invoke(prompt)
    return Command(
        goto=END,
        update={"messages": state["messages"] + [AIMessage(name="Writer", content=resp.content)]},
    )

# ---------- 4. Build the graph ----------
graph = StateGraph(GraphState)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)

graph.set_entry_point("researcher")
graph.add_edge("researcher", "writer")   # deterministic hand-off
graph.add_edge("writer", END)

#compiled = graph.compile(checkpointer=MemorySaver())  # resumable!
compiled = graph.compile()
  
# ---------- 5. Run it ----------
if __name__ == "__main__":
    user_question = "Explain how modern solar panels convert sunlight into electricity."
    result = compiled.invoke(
        {"messages": [HumanMessage(content=user_question)]}
    )
    print(result["messages"][-1].content)
