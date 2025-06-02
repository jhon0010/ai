from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv 
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate
)
from langchain.chains import ConversationChain


load_dotenv()

openai_model = "gpt-4o-mini"
llm = ChatOpenAI(temperature=0.0, model=openai_model)

"""_summary_
        Get chat history for a given session ID using InMemoryChatMessageHistory.
    Returns:
        _type_: _description_
"""
chat_map = {}
def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_map:
        chat_map[session_id] = InMemoryChatMessageHistory()
    return chat_map[session_id]

"""
ConversationBufferMemory is a memory class that stores the entire conversation.
It is a simple memory class that stores the entire conversation in a list.
"""
def conversation_buffer_memory():
    memory = ConversationBufferMemory(return_messages=True)
    memory.save_context(
        {"input": "Hi, my name is James"},  # user message
        {"output": "Hey James, what's up? I'm an AI model called Zeta."}  # AI response
    )
    memory.save_context(
        {"input": "I'm researching the different types of conversational memory."},  # user message
        {"output": "That's interesting, what are some examples?"}  # AI response
    )
    memory.save_context(
        {"input": "I've been looking at ConversationBufferMemory and ConversationBufferWindowMemory."},  # user message
        {"output": "That's interesting, what's the difference?"}  # AI response
    )
    memory.save_context(
        {"input": "Buffer memory just stores the entire conversation, right?"},  # user message
        {"output": "That makes sense, what about ConversationBufferWindowMemory?"}  # AI response
    )
    memory.save_context(
        {"input": "Buffer window memory stores the last k messages, dropping the rest."},  # user message
        {"output": "Very cool!"}  # AI response
    )
    print(memory.load_memory_variables({}))
    

def conversation_chain_memory():
    
    memory = ConversationBufferMemory(return_messages=True)
    memory.chat_memory.add_user_message("Hi, my name is James")
    memory.chat_memory.add_ai_message("Hey James, what's up? I'm an AI model called Zeta.")
    memory.chat_memory.add_user_message("I'm researching the different types of conversational memory.")
    memory.chat_memory.add_ai_message("That's interesting, what are some examples?")
    memory.chat_memory.add_user_message("I've been looking at ConversationBufferMemory and ConversationBufferWindowMemory.")
    memory.chat_memory.add_ai_message("That's interesting, what's the difference?")
    memory.chat_memory.add_user_message("Buffer memory just stores the entire conversation, right?")
    memory.chat_memory.add_ai_message("That makes sense, what about ConversationBufferWindowMemory?")
    memory.chat_memory.add_user_message("Buffer window memory stores the last k messages, dropping the rest.")
    memory.chat_memory.add_ai_message("Very cool!")
    memory.load_memory_variables({})  


    chain = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    response = chain.run("What is my name?")
    print(response)


def conversation_with_runnable_with_message_history():
    system_prompt = "You are a helpful assistant called Zeta."

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{query}"),
    ])
    from langchain_core.runnables.history import RunnableWithMessageHistory
    
    pipeline = prompt_template | llm
    pipeline_with_history = RunnableWithMessageHistory(
        pipeline,
        get_session_history=get_chat_history,
        input_messages_key="query",
        history_messages_key="history"
    )
    response = pipeline_with_history.invoke(
        {"session_id": "123", "query": "What is my name?"}
    )
    print(response.content)
    
"""

Summary memory is a type of memory that summarizes the conversation. Ypu cna see the tarce like Current conversation:
Instead of storing the entire conversation, it stores a summary of the conversation.
Useful on very long conversations where you don't want to store the entire conversation. save tokens!
"""    
def conversation_with_summary_memory():
    from langchain.memory import ConversationSummaryMemory
    memory = ConversationSummaryMemory(llm=llm)
    chain = ConversationChain(
        llm=llm,
        memory = memory,
        verbose=True
    )
    chain.invoke({"input": "hello there my name is James"})
    chain.invoke({"input": "I am researching the different types of conversational memory."})
    chain.invoke({"input": "I have been looking at ConversationBufferMemory and ConversationBufferWindowMemory."})
    chain.invoke({"input": "Buffer memory just stores the entire conversation"})
    chain.invoke({"input": "Buffer window memory stores the last k messages, dropping the rest."})
    
    response = chain.invoke({"input": "What is my name again?"})
    print(response)

"""
MAIN
"""
if __name__ == "__main__":
    #conversation_buffer_memory()
    #conversation_chain_memory()
    conversation_with_summary_memory()
