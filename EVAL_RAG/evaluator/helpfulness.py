from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from openevals.prompts import RAG_HELPFULNESS_PROMPT
from langchain_core.prompts import PromptTemplate


# Grade output schema
class HelpfulnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    helpfulness: Annotated[
        bool, ..., "Provide the score on whether the answer addresses the question"
    ]


# Grade prompt
helpfulness_prompt = PromptTemplate.from_template(RAG_HELPFULNESS_PROMPT)

# Grader LLM
helpfulness_llm = ChatOpenAI(
    model="gpt-4.1-mini", temperature=0
).with_structured_output(HelpfulnessGrade, method="json_schema", strict=True)

helpfulness_chain = helpfulness_prompt | helpfulness_llm


# Evaluator
def helpfulness(inputs: dict, outputs: dict) -> bool:
    """An evaluator for RAG answer helpfulness."""
    question = inputs.get("input")
    answer = outputs.get("answer")
    grade = helpfulness_chain.invoke({"inputs": question, "outputs": answer})
    return grade["helpfulness"]
