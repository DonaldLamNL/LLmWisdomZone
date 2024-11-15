import os
import time
from tenacity import (retry, stop_after_attempt, wait_random_exponential)

from llm.base import *
from llm.format import *
from utils.global_functions import *

from openai import AzureOpenAI

AZURE_OPENAI_API_KEY = os.getenv(f"AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv(f"AZURE_ENDPOINT")

client = AzureOpenAI(
    azure_endpoint = AZURE_ENDPOINT, 
    api_key=AZURE_OPENAI_API_KEY,  
    api_version="2023-03-15-preview",
)

class AzureChat(LLMChat):
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def get_msg(self, messages: List[Message]) -> List[dict]:
        return [msg.to_openai_format() for msg in messages]
        
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def chat(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        
        temperature = temperature if temperature else self.DEFAULT_TEMPERATURE
        max_tokens = max_tokens if max_tokens else self.DEFAULT_MAX_TOKENS
        num_comps = num_comps if num_comps else self.DEFAULT_NUM_COMPLETIONS
        inputs = self.get_msg(messages)
        
        response = client.chat.completions.create(
            model=self.model_name,
            messages=inputs,
            temperature=temperature,
        )
        
        write_records(response.choices[0].message.content, title="RESPONSE")
        time.sleep(self.DEFAULT_DELAY)
        
        return response.choices[0].message.content
    