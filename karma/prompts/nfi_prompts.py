"""User-prompt task templates for NFI datasets."""

from karma.prompts import Prompt

EKA_NFI_COT = Prompt(
    name="eka_nfi_cot",
    version=1,
    role="user",
    body="""
You are an expert at Indian Pharmacology.
Solve them in a step-by-step fashion. 
Output only the final answer (e.g., A, B, C, D, or E) 
Question: <QUESTION>
""",
)


