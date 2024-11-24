from llm.base import LLMChat
from utils.global_functions import *
from debator import LLMDebator, MLLMDebator, MainAgent

SYSTEM_PROMPT = "You are a helpful assistant."

class Debate():
    rounds: int = 5
    
    def __init__(self, llm, mllm):
        self.llm = llm
        self.mllm = mllm
        
    def run(self, **kwargs) -> str:
        self.llm_debator = LLMDebator(agent=self.llm, **kwargs)
        llm_reason = self.llm_debator.init_run()
    
        self.mllm_debator = MLLMDebator(agent=self.mllm, **kwargs)
        mllm_reason = self.mllm_debator.init_run()
        
        self.main_agent = MainAgent(agent=self.mllm, **kwargs)
        comments = llm_reason + "\n" + mllm_reason
        answer = self.main_agent.run(comments)
        
        return llm_reason, mllm_reason, answer
        
    