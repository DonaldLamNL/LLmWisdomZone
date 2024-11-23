import ast
from typing import List

from llm import LLMChat, Message
from utils.log import console_output
from utils.global_functions import *


class EntityExtractor:
    def __init__(self, model: LLMChat) -> None:
        self.model = model

    def extract_entities(self, question: str, caption: str) -> List[str]:
        # Get the input prompt from entities_prompt.txt
        prompt = get_prompt_template(
            filepath="./prompt_templates/entities_prompt.txt",
            inputs=[question, caption]
        )

        # Format the model input
        inputs = [Message(role="system", content=SYSTEM_PROMPT)]
        inputs.append(Message(role="user", content=prompt))

        # GPT request
        for i in range(MAX_RETRY):
            # try:
            print("inputs: ", inputs)
            response = self.model.chat(inputs)
            print("response: ", response)
            if "extracted entities:" in response.lower():
                entities = response.lower().split("extracted entities:")[1].strip().split(", ")
            else:
                entities = []
            # entities_str = extract_content(response, "entities")
            # entities = ast.literal_eval(entities_str)
            print("entities: ", entities)
            return entities
            # except Exception as e:
            #     console_output(f"Retry Entity Extraction {i}: {e}")
        return []
    
