from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from openevals.prompts import CORRECTNESS_PROMPT
from langchain_core.prompts import PromptTemplate


### Correctness Evaluator
class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]


# Grade prompt
correctness_prompt = PromptTemplate.from_template(CORRECTNESS_PROMPT)
# Grader LLM
correctness_llm = ChatOpenAI(
    model="gpt-4.1-mini", temperature=0
).with_structured_output(CorrectnessGrade, method="json_schema", strict=True)

correctness_chain = correctness_prompt | correctness_llm


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """An Evaluator for RAG answer accuracy"""

    question = inputs.get("input")
    answer = outputs.get("answer")
    reference_answer = reference_outputs.get("output")
    grade = correctness_chain.invoke(
        {"inputs": question, "outputs": answer, "reference_outputs": reference_answer}
    )

    return grade["correct"]
