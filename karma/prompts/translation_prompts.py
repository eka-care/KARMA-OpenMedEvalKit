"""User-prompt bodies for translation datasets."""

from karma.prompts import Prompt

IN22CONV_TRANSLATE = Prompt(
    name="in22conv_translate",
    version=1,
    role="user",
    body="""Translate the given text to the target language. Output only the translation without any additional text.""",
)
