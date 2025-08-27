from typing import List

from typing_extensions import TypedDict


class CragState(TypedDict):

    question: str
    generation: str
    web_search: str
    documents: List[str]