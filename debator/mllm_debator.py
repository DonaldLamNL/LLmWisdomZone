import json
from llm.base import *
from utils.global_functions import *

SYSTEM_PROMPT = "You are a helpful assistant."

class MLLMDebator():
    agent: LLMChat
    question: str
    answer: str
    image_paths: str
    
    def __init__(self, agent, question, answer, image_paths) -> None:
        self.agent = agent
        self.question = question
        self.answer = answer
        self.image_paths = image_paths
        
    def init_run(self):
        self.history = [Message("system", SYSTEM_PROMPT)]
        input_prompt = get_prompt_template(
            filepath="./prompt_templates/mllm_debater.txt",
            inputs=[self.question, self.answer]
        )
        self.history.append(Message(role="user", content=input_prompt, image_paths=self.image_paths))
        
        for i in range(MAX_RETRY):
            try:
                response = self.agent.chat(self.history)
                return response
                stand, reason = extract_content(response, "stand"), extract_content(response, "reason")
                # json_response = extract_json_format(response)
                # json_response = json.loads(json_response)
                # return json_response["stand"], json_response["reason"]
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