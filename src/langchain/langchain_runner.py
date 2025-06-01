import requests
import uuid

from dotenv import load_dotenv 

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableParallel
from pydantic import BaseModel, Field
from PIL import Image
from io import BytesIO

from langsmith import traceable # Go to https://smith.langchain.com/ to get your API key

article = """
Cats are small, carnivorous mammals that are often kept as pets. They belong to the family Felidae and are known for their agility, sharp retractable claws, and keen senses. Cats have been domesticated for thousands of years and are one of the most popular pets worldwide.
"""

class ArticleResponse(BaseModel):
    content: str = Field(description="The original article text.")
    summary: str = Field(description="A summary of the article.")
    body: str = Field(description="The main body of the article.")
    call_out: str = Field(description="A call-out section for the article.")
    
 
@traceable(name="Image Generation")
def generate_and_save_image(image_prompt):
    try:
        dalle = DallEAPIWrapper(
            model="dall-e-3",
            size="1024x1024",
            quality="standard",   # or "hd"
        )
        image_url = dalle.run(image_prompt)
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        output_path = "generated_image" + uuid.uuid4().hex + ".png"
        image.save(output_path)
        print(f"Image has been saved to: {output_path}")
        return image_url
    except Exception as e:
            print(f"Error generating image: {str(e)}")

image_gen_runnable = RunnableLambda(generate_and_save_image)

@traceable(name="Print Model Structure")
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

    chain_two = (
        chain_one
        | RunnableLambda(lambda x: x.content)
        | image_gen_runnable
    )
    chain_two.invoke({"article": article})


if __name__ == "__main__":
    main()