from langchain_core.tools import tool
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, prompt
from langchain.agents import AgentExecutor
from dotenv import load_dotenv 
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents.react.agent import create_react_agent 
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage


load_dotenv()
model_name="gpt-4o-mini"
llm = ChatOpenAI(
    model_name=model_name,
    temperature=0.0,
)

"""
With the @tool decorator our function is turned into a StructuredTool object
"""
@tool
def add(x: float, y: float) -> float:
    """Add 'x' and 'y'."""
    return x + y

@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' and 'y'."""
    return x * y

@tool
def exponentiate(x: float, y: float) -> float:
    """Raise 'x' to the power of 'y'."""
    return x ** y

@tool
def subtract(x: float, y: float) -> float:
    """Subtract 'x' from 'y'."""
    return y - x

print(f"{add.name=}\n{add.description=}")

def see_tools_information():
    print(f"{add.name=}\n{add.description=}")   
    print(f"{multiply.name=}\n{multiply.description=}")
    print(f"{exponentiate.name=}\n{exponentiate.description=}")
    print(f"{subtract.name=}\n{subtract.description=}")
    print(f"{add.args_schema.model_json_schema}")

def defining_json_output():
    llm_output_string = "{\"x\": 5, \"y\": 2}"  # this is the output from the LLM
    llm_output_dict = json.loads(llm_output_string)  # load as dictionary
    llm_output_dict
    exponentiate.func(**llm_output_dict)  # call the function with the dictionary
    add.func(**llm_output_dict)  # call the function with the dictionary
    subtract.func(**llm_output_dict)  # call the function with the dictionary
    multiply.func(**llm_output_dict)  # call the function with the dictionary


def agent_calcculator() -> AgentExecutor:

    memory = ConversationBufferMemory(
        memory_key="chat_history",  # must align with MessagesPlaceholder variable_name
        return_messages=True  # to return Message objects
    )
    from langchain.agents import create_tool_calling_agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful calculator agent."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    tools = [add, subtract, multiply, exponentiate]
    agent = create_tool_calling_agent(
        llm=llm, tools=tools, prompt=prompt
    )
    agent.invoke({
        "input": "what is 10.7 multiplied by 7.68?",
        "chat_history": memory.chat_memory.messages,
        "intermediate_steps": []  # agent will append it's internal steps here
    })
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True
    )
    agent_executor.invoke({
        "input": "what is 10.7 multiplied by 7.68?",
        "chat_history": memory.chat_memory.messages,
    })
    print(memory.chat_memory.messages)
    return agent
    
    

def agent_with_duck_search():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
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
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}'''

    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    response = agent_executor.invoke({
        "input": "Look at he internet and help me to get the last new on Colombia",
        "messages": [],
        "agent_scratchpad": ""
    })
    print("Response = ",response)
       

if __name__ == "__main__":
    defining_json_output()
    see_tools_information()
    agent_calcculator()
    agent_with_duck_search()