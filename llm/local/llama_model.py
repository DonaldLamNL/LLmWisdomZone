from llm.base import *
from llm.format import *
from utils.global_functions import *

import transformers
import torch

class LlamaChat(LLMChat):
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", device="cuda"):
        # Initialize the processor and model with the specified model name and device
        self.model = transformers.pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        # self.device = device
        # self.model.to(self.device)

    def chat(self, messages: List[Message]) -> Union[List[str], str]:
        conversation = [message.to_llama_format() for message in messages]

        # Apply the chat template and process the inputs
        outputs = self.model(conversation, max_new_tokens=256)
        result = outputs[0]["generated_text"][-1]['content']
        return result