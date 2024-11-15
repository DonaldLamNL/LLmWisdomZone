import os
from abc import ABC, abstractmethod
from typing import List, Union, Optional

from llm.format import *

class LLMChat(ABC):
    DEFAULT_MAX_TOKENS = 1024
    DEFAULT_TEMPERATURE = 0
    DEFAULT_NUM_COMPLETIONS = 1
    DEFAULT_DELAY = 0

    # @abstractmethod
    def chat(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:

        pass

    def get_msg(self):
        pass


def write_records(message: str, title: str = "LOGS", file: str = "logs") -> None:
    os.makedirs(file, exist_ok=True)
    with open(f"./{file}/records.txt", "a") as record_file:
        if title == "LOGS":
            record_file.write(f"================================ {title} ================================\n")
        else:
            record_file.write(f"-------------------------------- {title} --------------------------------\n")
        record_file.write(str(message) + "\n")