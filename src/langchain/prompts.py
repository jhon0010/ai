from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv 

load_dotenv()

openai_model = "gpt-4o-mini"
llm = ChatOpenAI(temperature=0.0, model=openai_model)

"""
The below is an example of how a RAG prompt may look:

Answer the question based on the context below,                 }
if you cannot answer the question using the                     }--->  (Rules) For Our Prompt
provided information answer with "I don't know"                 }

Context: Aurelio AI is an AI development studio                 }
focused on the fields of Natural Language Processing (NLP)      }
and information retrieval using modern tooling                  }--->   Context AI has
such as Large Language Models (LLMs),                           }
vector databases, and LangChain.                                }

Question: Does Aurelio AI do anything related to LangChain?     }--->   User Question

Answer:                                                         }--->   AI Answer
"""
def answer_question_based_on_context():

        prompt = """
        Answer the user's query based on the context below.
        If you cannot answer the question using the
        provided information answer with "I don't know".

        Context: {context}
        """
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", prompt),
            ("user", "{query}"),
        ])
    

        # Explore the input variables and messages
        prompt_template.input_variables
        prompt_template.messages

        pipeline = (
            {
                "query": lambda x: x["query"],
                "context": lambda x: x["context"]
            }
            | prompt_template
            | llm
        )

        context = """Aurelio AI is an AI company developing tooling for AI
        engineers. Their focus is on language AI with the team having strong
        expertise in building AI agents and a strong background in
        information retrieval.

        The company is behind several open source frameworks, most notably
        Semantic Router and Semantic Chunkers. They also have an AI
        Platform providing engineers with tooling to help them build with
        AI. Finally, the team also provides development services to other
        organizations to help them bring their AI tech to market.

        Aurelio AI became LangChain Experts in September 2024 after a long
        track record of delivering AI solutions built with the LangChain
        ecosystem."""

        query = "what does Aurelio AI do?"
        response = pipeline.invoke({"query": query, "context": context})
        print(response)
        
"""
Few shot prompt example is a technique to improve the performance of the model.
It is a technique to provide the model with a few examples of the expected output.
The model will then use these examples to generate the output for the new input.

For example, if we want to generate an article about cat behavior, 
we can provide the model with a few examples of the expected output.
"""
def few_shot_prompt():
    from langchain.prompts import FewShotChatMessagePromptTemplate
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])
    examples = [
        {
            "input": "Create an article about cat behavior",
            "output": "Title: Understanding Feline Behavior\n\nSummary: A comprehensive guide to understanding why cats behave the way they do, from their hunting instincts to their social interactions.\n\nBody: Cats are fascinating creatures with complex behaviors rooted in their evolutionary history. Their hunting instincts, territorial nature, and social dynamics all contribute to their unique personalities...\n\nCall-out: Did you know? Cats spend 70% of their day sleeping, which is why they're so active at night!"
        },
        {
            "input": "Write about cat health and care",
            "output": "Title: Essential Cat Care Guide\n\nSummary: Everything you need to know about keeping your feline friend healthy and happy, from nutrition to regular check-ups.\n\nBody: Proper cat care involves a combination of good nutrition, regular veterinary visits, and understanding your cat's needs. A balanced diet, clean litter box, and regular playtime are essential...\n\nCall-out: Regular veterinary check-ups can help catch health issues early and extend your cat's life by up to 3 years!"
        },
        {
            "input": "Explain cat communication",
            "output": "Title: Decoding Cat Communication\n\nSummary: Learn to understand your cat's body language, vocalizations, and behavioral cues to strengthen your bond.\n\nBody: Cats communicate through a complex system of body language, vocalizations, and scent marking. From the position of their ears to the movement of their tail, every gesture has meaning...\n\nCall-out: A cat's purr isn't just a sign of happiness - it can also indicate stress or pain, making it crucial to understand the context!"
        }
    ]
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    # here is the formatted prompt
    print(few_shot_prompt.format())
    
def chain_of_thought_prompt():
    query = (
        "How many keystrokes are needed to type the numbers from 1 to 500?"
    )
    # Define the chain-of-thought prompt template
    cot_system_prompt = """
    Be a helpful assistant and answer the user's question.

    To answer the question, you must:

    - List systematically and in precise detail all
    subproblems that need to be solved to answer the
    question.
    - Solve each sub problem INDIVIDUALLY and in sequence.
    - Finally, use everything you have worked through to
    provide the final answer.
    """

    cot_prompt_template = ChatPromptTemplate.from_messages([
        ("system", cot_system_prompt),
        ("user", "{query}"),
    ])

    cot_pipeline = cot_prompt_template | llm
    cot_response = cot_pipeline.invoke({"query": query})
    print(cot_response)
    
if __name__ == "__main__":
    answer_question_based_on_context()
    few_shot_prompt()
    chain_of_thought_prompt()