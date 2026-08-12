"""User-prompt task templates for visual medical QA datasets."""

from karma.prompts import Prompt

SLAKE_COT = Prompt(
    name="slake_cot",
    version=1,
    role="user",
    body="""<QUESTION> You may write out your argument before stating your final very short,
definitive, and concise answer (if possible, a single word or the letter corresponding to your answer
choice) X in the format "Final Answer: X": """,
)

VQA_RAD_COT = Prompt(
    name="vqa_rad_cot",
    version=1,
    role="user",
    body="""Given this radiology image, which can be a frontal chest X-ray, a single slice head or
abdominal CT or MR image, provide a very short, definitive, and concise answer (if possible, a single
word) to the following question: <QUESTION>""",
)
