from typing import List
from typing_extensions import TypedDict

class SelfState(TypedDict):
    retry_count : int
    question: str
    generation: str
    documents: List[str]