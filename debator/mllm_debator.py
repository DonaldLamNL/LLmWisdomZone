import json
from llm.base import *
from utils.global_functions import *

SYSTEM_PROMPT = "You are a helpful assistant."

class MLLMDebator():
    def __init__(self, agent, **kwargs) -> None:
        self.agent = agent
        self.image_path = kwargs.get("image_path", "No image path provided")
        self.question = kwargs.get("question", "No question provided")
        self.caption = kwargs.get("caption", "No caption provided")
        self.subquestions = kwargs.get("subquestions", "No subquestions provided")
        self.subquestions_answer = kwargs.get("subquestions_answer", "No answer provided")
        self.summarization = kwargs.get("summarization", "No summarization provided")
        self.final_answer = kwargs.get("final_answer", "No final answer provided")

        
    def init_run(self):
        self.history = [Message("system", SYSTEM_PROMPT)]
        input_prompt = get_prompt_template(
            filepath="./prompt_templates/mllm_debater.txt",
            inputs=[
                self.question,
                self.caption,
                self.subquestions,
                self.subquestions_answer,
                self.summarization,
                self.final_answer,
            ]
        )
        self.history.append(Message(role="user", content=input_prompt, image_paths=[self.image_path]))
        
        for i in range(MAX_RETRY):
            try:
                response = self.agent.chat(self.history)
                return response
            except:
                print(f"Generate: Retry {i+1}")
    
    
    def run(self, response:str) -> str:
        self.history.append(Message("user", f"Response from agent: {response}"))
        for i in range(MAX_RETRY):
            try:
                response = self.agent.chat(self.history)
                return response

                json_response = extract_json_format(response)
                self.history.append(Message("assistant", str(json_response)))
                return json_response["pass"], json_response["reason"]
            except:
                print(f"Generate sub-questions: Retry {i+1}")