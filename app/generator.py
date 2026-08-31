from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from app.config import config

def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=config["groq_api_key"],
    )

def evaluate_context(question: str, context: str, llm) -> bool:
    prompt = PromptTemplate.from_template("""
You are evaluating whether a retrieved context is sufficient to answer a question.
Question: {question}
Retrieved Context: {context}
Is the context sufficient to give a complete and accurate answer?
Reply with ONLY one word: YES or NO.
""")
    chain = prompt | llm
    response = chain.invoke({"question": question, "context": context})
    return response.content.strip().upper().startswith("YES")

def reformulate_query(question: str, context: str, llm) -> str:
    prompt = PromptTemplate.from_template("""
The following question could not be answered with the retrieved context.
Reformulate the question to search for more specific information.
Original Question: {question}
Retrieved Context (insufficient): {context}
Write a better search query (one line only, no explanation):
""")
    chain = prompt | llm
    response = chain.invoke({"question": question, "context": context})
    return response.content.strip()

def generate_answer(question: str, context: str, llm) -> str:
    prompt = PromptTemplate.from_template("""
You are an HR assistant. Answer the question using only the context below.
If the answer is not in the context, say "I don't have enough information."
Context:
{context}
Question: {question}
Answer:""")
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content
