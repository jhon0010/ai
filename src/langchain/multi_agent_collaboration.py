from __future__ import annotations
from typing import TypedDict, Literal, List, Annotated
from uuid import uuid4

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents.react.agent import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor
 
from IPython.display import Image

class GraphState(TypedDict):
    messages: List[AIMessage | HumanMessage]
    status: Literal["pending", "ready", "more_information"]


llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)


def duck_search(state: GraphState) -> Command[Literal["researcher"]]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful internet search assitant help me find information on a topic."),
        ("human", "{input}"),
    ])
    search = DuckDuckGoSearchRun()
    tools = [search]

    template = '''Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times) use {agent_scratchpad}
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    '''

    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True ,verbose=True)
    response = agent_executor.invoke({
        "input": state["messages"][-1].content,
        "messages": state["messages"]
    })
    print("Response = ", response)
    content = response.get("output", str(response))
    return Command(
        goto="researcher",
        update={
            "messages": state["messages"] + [AIMessage(name="Internet Searcher", content=content)],
            "status": "pending"
        },
    )

def researcher(state: GraphState) -> Command[Literal["writer"]]:
    """Pull out the latest user question and return research bullets."""
    user_query = state["messages"][-1].content
    prompt = (
        "You're a researcher. Decide if the current information is enough.\n"
        "If YES, answer with 'ready' and produce concise, referenced bullet-points it.\n"
        "If NO, answer with 'more_information' and explain what is missing.\n\n"
        f"Current info:\n{user_query}"
    )
    resp = llm.invoke(prompt)
    if "more_information" in resp.content.lower():
        next_node = "duck_search"
        status = "more_information"
    else:
        next_node = "writer"
        status = "ready"
    return Command(
        goto=next_node,
        update={
            "messages": state["messages"] + [AIMessage(name="Researcher", content=resp.content)],
            "status": status,
        },
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
        update={"messages": state["messages"] + [AIMessage(name="Writer", content=resp.content)],
                 "status": "done"},
    )

def create_graph() -> CompiledStateGraph:
    """Create the state graph for the multi-agent collaboration."""
    graph = StateGraph(GraphState)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("duck_search", duck_search)

    graph.set_entry_point("duck_search")
    graph.add_edge("duck_search", "researcher") 
    graph.add_edge("researcher", "writer")
    graph.add_edge("researcher", "duck_search") 
    graph.add_edge("writer", END)
    compiled = graph.compile(checkpointer=MemorySaver()) 
    img = Image(compiled.get_graph().draw_mermaid_png())
    with open(f"multi_agent.png", "wb") as f:
        f.write(img.data)
    return compiled


if __name__ == "__main__":
    user_question = "Explain how modern solar panels convert sunlight into electricity."
    compiled_graph = create_graph() 
    result = compiled_graph.invoke(
        {"messages": [HumanMessage(content=user_question)]},
        {"configurable": {"thread_id": "test_123"}}
    )
    print(result["messages"][-1].content)
