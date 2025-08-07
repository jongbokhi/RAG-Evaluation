from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from openevals.prompts import RAG_GROUNDEDNESS_PROMPT
from langchain_core.prompts import PromptTemplate


# Grade output schema
class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[
        bool, ..., "Provide the score on if the answer hallucinates from the documents"
    ]


grounded_prompt = PromptTemplate.from_template(RAG_GROUNDEDNESS_PROMPT)
# Grader LLM
grounded_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0).with_structured_output(
    GroundedGrade, method="json_schema", strict=True
)

chain = grounded_prompt | grounded_llm


def groundedness(inputs: dict, outputs: dict) -> bool:
    """An evaluator for RAG answer groundedness."""

    context = "\n\n".join(doc.page_content for doc in outputs.get("documents"))
    answer = outputs.get("answer")
    grade = chain.invoke({"context": context, "outputs": answer})
    return grade["grounded"]
