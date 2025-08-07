from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from openevals.prompts import RAG_RETRIEVAL_RELEVANCE_PROMPT
from langchain_core.prompts import PromptTemplate


# Grade output schema
class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    retrieval_relevance: Annotated[
        bool,
        ...,
        "True if the retrieved documents are relevant to the question, False otherwise",
    ]


# Grade prompt
retrieval_relevance_prompt = PromptTemplate.from_template(
    RAG_RETRIEVAL_RELEVANCE_PROMPT
)

# Grader LLM
retrieval_relevance_llm = ChatOpenAI(
    model="gpt-4.1-mini", temperature=0
).with_structured_output(RetrievalRelevanceGrade, method="json_schema", strict=True)

retrieval_relevance_chain = retrieval_relevance_prompt | retrieval_relevance_llm


def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    """An evaluator for document relevance"""
    input = inputs.get("input")
    retrieved_docs = "\n\n".join(doc.page_content for doc in outputs.get("documents"))

    # Run evaluator
    grade = retrieval_relevance_chain.invoke(
        {"inputs": input, "context": retrieved_docs}
    )
    return grade["retrieval_relevance"]
