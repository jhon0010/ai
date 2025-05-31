from dotenv import load_dotenv 

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableParallel

article = """
Cats are small, carnivorous mammals that are often kept as pets. They belong to the family Felidae and are known for their agility, sharp retractable claws, and keen senses. Cats have been domesticated for thousands of years and are one of the most popular pets worldwide.
"""

"""
This script demonstrates how to create a LangChain agent that can process an article and respond to questions about it.
"""
def main() -> None:
    load_dotenv()
    print("Starting LangChain agent...")
    system_pompt = SystemMessagePromptTemplate.from_template(
        """
        You are a helpful AI assistant. Your task is to answer questions based on the provided article. 
        If you don't know the answer, say 'I don't know'. 
        Here is the article: 
            ---------
                {article}
            ---------
        """
    )
    system_pompt = system_pompt.format(article=article)


    human_prompt = HumanMessagePromptTemplate.from_template(
        """
        Your task is to help me to create a complete article with the following pieces of information: 
            tittle, summary_and_captive_introduction, body, call_out.
        Use as context the following article: 
            ---------
                {article}
            ---------
        """,
        input_variables=["article"]
    )
    human_prompt = human_prompt.format(article=article)

    main_prompt = ChatPromptTemplate.from_messages(
        [system_pompt, human_prompt]
    )
    print("System prompt and human prompt created successfully." + main_prompt.format(article=article))
    
    openai_model = "gpt-4o-mini"
    llm = ChatOpenAI(temperature=0.0, model=openai_model)
    creative_llm = ChatOpenAI(temperature=0.9, model=openai_model)
    
    inputs = RunnableParallel({
        "input": RunnableLambda(lambda _: article)
    })
    chain_one = (
        inputs
        | main_prompt
        | creative_llm
    )
    response = chain_one.invoke({"article": article})
    print("Response from the agent: " + response.get("text", ""))


"""
    RUN ME!
"""
if __name__ == "__main__":
    main()