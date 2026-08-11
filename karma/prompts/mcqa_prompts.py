"""User-prompt (task-template) bodies for multiple-choice medical QA datasets.

Each is the confinement instruction that wraps a question; ``<QUESTION>`` is
substituted per-sample by the dataset's ``format_item``.
"""

from karma.prompts import Prompt

MEDMCQA_COT = Prompt(
    name="medmcqa_cot",
    version=1,
    role="user",
    body="""Instructions: The following are multiple choice questions about medical knowledge. Solve them in a
step-by-step fashion, starting by summarizing the available information. Output a single option from the
four options as the final answer. Question: <QUESTION> Response (think step by step and then
end with "Final Answer:" followed by *only* the letter corresponding to the correct answer enclosed in
parentheses)""",
)

MEDMCQA_INDIC_COT = Prompt(
    name="medmcqa_indic_cot",
    version=1,
    role="user",
    body="""Instructions: The following are multiple choice questions about medical knowledge. Solve them in a
step-by-step fashion, starting by summarizing the available information. Output a single option from the
four options as the final answer. Question: <QUESTION> Response (think step by step and then
end with "Final Answer:" followed by *only* the letter corresponding to the correct answer enclosed in
parentheses)""",
)

MEDQA_COT = Prompt(
    name="medqa_cot",
    version=1,
    role="user",
    body="""Instructions: The following are multiple choice questions about medical knowledge. Solve them in a
step-by-step fashion, starting by summarizing the available information. Output a single option from the
four options as the final answer. Question: <QUESTION> Response (think step by step and then
end with "Final Answer:" followed by *only* the letter corresponding to the correct answer enclosed in
parentheses)""",
)

PUBMEDMCQA_COT = Prompt(
    name="pubmedmcqa_cot",
    version=1,
    role="user",
    body="""Instructions: The following are multiple choice questions about medical knowledge. Solve them in a
step-by-step fashion, starting by summarizing the available information. Output a single option from the
four options as the final answer. Answer the following question given the context (reply with one of the
options): Context: <CONTEXT> Question: <QUESTION> Response (think step by step and
then end with "Final Answer:" followed by *only* the letter corresponding to the correct answer enclosed
in parentheses)""",
)

NFIMCQA_COT = Prompt(
    name="nfimcqa_cot",
    version=1,
    role="user",
    body="""Instructions: The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion. Output only the final answer (e.g., A, B, C, or D) at the end using the format "Final Answer: (X)". Question: <QUESTION>""",
)

MMLU_MEDICAL_COT = Prompt(
    name="mmlu_medical_cot",
    version=1,
    role="user",
    body="""Instructions: The following are multiple choice questions about medical knowledge. Solve them in a
step-by-step fashion, starting by summarizing the available information. Output a single option from the
four options as the final answer. Question: <QUESTION> Response (think step by step and then
end with "Final Answer:" followed by *only* the letter corresponding to the correct answer enclosed in
parentheses)""",
)

EKA_MMLU_MEDICAL_COT = Prompt(
    name="eka_mmlu_medical_cot",
    version=1,
    role="user",
    body="""Instructions: The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion, starting by summarizing the available information. Output a single option from the four options as the final answer. Question: <QUESTION> Response (think step by step and then end with "Final Answer:" followed by *only* the letter corresponding to the correct answer enclosed in parentheses)""",
)

INDIAN_DRUG_MCQA_COT = Prompt(
    name="indian_drug_mcqa_cot",
    version=1,
    role="user",
    body="""Instructions: The following are multiple choice questions about Indian pharmaceutical medications. Solve them in a step-by-step fashion. Output only the final answer (e.g., A, B, C, D, or E) at the end using the format "Final Answer: (X)". Question: <QUESTION>""",
)

MEDXPERTQA_COT = Prompt(
    name="medxpertqa_cot",
    version=1,
    role="user",
    body="""<QUESTION> Think
step by step through each of the multiple choice options. You MUST end your response with "Final
Answer:" followed by only the letter corresponding to the correct answer enclosed in parentheses).""",
)

MEDXPERTQA_TEXT_COT = Prompt(
    name="medxpertqa_text_cot",
    version=1,
    role="user",
    body="""<QUESTION> Think
step by step through each of the multiple choice options. You MUST end your response with "Final
Answer:" followed by only the letter corresponding to the correct answer enclosed in parentheses).""",
)

MEDQA_USMLE_COT = Prompt(
    name="medqa_usmle_cot",
    version=1,
    role="user",
    body="""Instructions: The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion, starting by summarizing the available information. Output a single option from the four options as the final answer. Question: <QUESTION> Response (think step by step and then end with "Final Answer:" followed by *only* the letter corresponding to the correct answer enclosed in parentheses)""",
)


