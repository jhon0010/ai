from dotenv import load_dotenv 

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableParallel
from pydantic import BaseModel, Field
from dataclasses import asdict

class ArticleResponse(BaseModel):
    content: str = Field(description="The original article text.")
    summary: str = Field(description="A summary of the article.")
    body: str = Field(description="The main body of the article.")
    call_out: str = Field(description="A call-out section for the article.")


article = """
Cats are small, carnivorous mammals that are often kept as pets. They belong to the family Felidae and are known for their agility, sharp retractable claws, and keen senses. Cats have been domesticated for thousands of years and are one of the most popular pets worldwide.
"""

"""
This script demonstrates how to create a LangChain agent that can process an article and respond to questions about it.
"""

def print_model_structure(response: ArticleResponse) -> None:   
    print("\nModel Structure:")
    print("Type:", type(response))
    print("\nFields:")
    for field_name, field_info in response.model_fields.items():
        print(f"\n{field_name}:")
        print(f"  Type: {field_info.annotation}")
        print(f"  Required: {field_info.is_required()}")
        print(f"  Default: {field_info.default}")
        try:
            value = getattr(response, field_name)
            print(f"  Value: {value}")
        except Exception as e:
            print(f"  Value: <Error: {str(e)}>")


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
    ).format(article=article)


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
    ).format(article=article)

    main_prompt = ChatPromptTemplate.from_messages(
        [system_pompt, human_prompt]
    )
    openai_model = "gpt-4o-mini"
    creative_llm = ChatOpenAI(temperature=0.9, model=openai_model)
    creative_llm = creative_llm.with_structured_output(ArticleResponse)
    
    inputs = RunnableParallel({
        "input": RunnableLambda(lambda _: article)
    })
    chain_one = (
        inputs
        | main_prompt
        | creative_llm
    )
    response = chain_one.invoke({"article": article})
    print_model_structure(response)


"""
    RUN ME!
"""
if __name__ == "__main__":
    main()