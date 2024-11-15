import json
from llm.base import *
from utils.global_functions import *

SYSTEM_PROMPT = "You are a helpful assistant."

class MainAgent():
    agent: LLMChat
    question: str
    answer: str
    summarized: str
    image_paths: str
    
    def __init__(self, agent, question, answer, summarized, image_paths) -> None:
        self.agent = agent
        self.question = question
        self.answer = answer
        self.summarized = summarized
        self.image_paths = image_paths
        self.history = [Message("system", SYSTEM_PROMPT)]
    
    
    def run(self, response:str) -> str:
        input_prompt = get_prompt_template(
            filepath="./prompt_templates/main_agent_debate.txt",
            inputs=[self.question, self.summarized, self.answer, response]
        )
        self.history.append(Message("user", input_prompt, self.image_paths))
        for i in range(MAX_RETRY):
            try:
                response = self.agent.chat(self.history)
                return response

                stand, reason, answer = extract_content(response, "stand"), extract_content(response, "reason"), extract_content(response, "answer")
                return stand, reason, answer
                # json_response = extract_json_format(response)
                # json_response = json.loads(json_response)
                # return json_response["stand"], json_response["reason"], json_response["answer"]
            except:
                print(f"Generate sub-questions: Retry {i+1}")