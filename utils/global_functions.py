import re
import ast
from typing import List, Tuple
from llm.base import LLMChat
from llm.format import Message

SYSTEM_PROMPT = "You are a helpful assistant."
MAX_RETRY=5

# Get Prompt Template
def get_prompt_template(filepath: str, inputs: List) -> str:
    with open(filepath, 'r') as file:
        generated_prompt = file.read().split("<commentblockmarker>###</commentblockmarker>")[1].strip()
    for index, item in enumerate(inputs):
        key = f"!<INPUT {index+1}>!"
        generated_prompt = generated_prompt.replace(key, str(item))
    return generated_prompt


def extract_pattern(content: str, regex: str) -> Tuple[str, List[str]]:
    """
    Extract content that matches the given regex pattern from the input content,
    and return the content without the matched parts and the list of matched content.
    """
    # Find all matches of the pattern and remove the matched content from the content
    matched_content = re.findall(regex, content, re.DOTALL | re.MULTILINE)
    remove_content = re.sub(regex, "", content, flags=re.DOTALL | re.MULTILINE).strip()
    return remove_content.strip(), [c.strip() for c in matched_content]


def extract_content(content: str, key: str) -> str:
    """
    Extract the content between the special tokens: [START/{key}] and [END/{key}]
    """
    # Use f-string to build the regex pattern with the provided key
    pattern = rf"\[START/{key}\](.*?)\[END/{key}\]"
    _, extracted_content = extract_pattern(content, pattern)

    # Return the first match if available, otherwise None
    return extracted_content[0] if extracted_content else None


def generate_caption(model, image_paths):
    inputs = [Message(role="user", content="Generate a short caption to describe the image.", image_paths=image_paths)]
    for i in range(MAX_RETRY):
        try:
            response = model.chat(inputs)
            return response
        except:
            print(f"Generate sub-questions: Retry {i+1}")


def subquestion_generator(entity: str, question: str, caption:str, model: LLMChat) -> List[dict]:
    input_prompt = get_prompt_template(
        filepath="./prompt_templates/subquestion_prompt.txt",
        inputs=[entity, question, caption]
    )
    inputs = [Message(role="system", content=SYSTEM_PROMPT)]
    inputs.append(Message(role="user", content=input_prompt))
    for i in range(MAX_RETRY):
        try:
            response = model.chat(inputs)
            subquestions = response.lower().split("sub-questions:")[1].strip().split(", ")
            return [{"question": question} for question in subquestions]
        except:
            print(f"Generate sub-questions: Retry {i+1}")
            

def subquestion_answering(image_paths:str, caption:str, question: str, entity: str, model: LLMChat) -> str:
    input_prompt = get_prompt_template(
        filepath="./prompt_templates/subquestion_answering.txt",
        inputs=[caption, question, entity]
    )
    inputs = [Message(role="system", content=SYSTEM_PROMPT)]
    inputs.append(Message(role="user", content=input_prompt, image_paths=image_paths))
    for i in range(MAX_RETRY):
        try:
            response = model.chat(inputs)
            return response
        except:
            print(f"Generate sub-questions: Retry {i+1}")


def summarization(entity:str, pairs:List[dict[str]], model: LLMChat) -> str:
    information = "\n".join([f"{p['question']}\n{p['answer']}" for p in pairs])
    input_prompt = get_prompt_template(
        filepath="./prompt_templates/summarization.txt",
        inputs=[entity, information]
    )
    inputs = [Message(role="system", content=SYSTEM_PROMPT)]
    inputs.append(Message(role="user", content=input_prompt))
    for i in range(MAX_RETRY):
        try:
            response = model.chat(inputs)
            return response
        except:
            print(f"Generate sub-questions: Retry {i+1}")


def question_answering(image_paths:str, question: str, information:str, model: LLMChat) -> List[dict]:
    input_prompt = get_prompt_template(
        filepath="./prompt_templates/question_answering.txt",
        inputs=[question, information]
    )
    inputs = [Message(role="system", content=SYSTEM_PROMPT)]
    inputs.append(Message(role="user", content=input_prompt, image_paths=image_paths))
    for i in range(MAX_RETRY):
        try:
            response = model.chat(inputs)
            return response
        except:
            print(f"Generate sub-questions: Retry {i+1}")