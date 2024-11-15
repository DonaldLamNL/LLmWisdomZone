from llm.base import LLMChat
from utils.global_functions import *
from debator import LLMDebator, MLLMDebator, MainAgent

SYSTEM_PROMPT = "You are a helpful assistant."

class Debate():
    rounds: int = 5
    
    def __init__(self, llm, mllm):
        self.llm = llm
        self.mllm = mllm
        
    def run(self, image_paths:str, question:str, answer:str, summarized:str) -> str:
        # for i in range(self.rounds):
        llm_reason = ""
        if summarized != "":
            self.llm_debator = LLMDebator(agent=self.llm, question=question, answer=answer, summarized=summarized)
            llm_reason = self.llm_debator.init_run()
    
        self.mllm_debator = MLLMDebator(agent=self.mllm, question=question, answer=answer, image_paths=image_paths)
        mllm_reason = self.mllm_debator.init_run()
        
        self.main_agent = MainAgent(agent=self.mllm, question=question, answer=answer, summarized=summarized, image_paths=image_paths)
        comments = llm_reason + "\n" + mllm_reason
        answer = self.main_agent.run(comments)
        
        return answer
        
    