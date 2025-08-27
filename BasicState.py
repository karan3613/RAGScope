from typing_extensions import TypedDict


class BasicState(TypedDict):
    generation : str
    question : str
    documents : str