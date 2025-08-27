from typing_extensions import TypedDict


class AdaptiveState(TypedDict):
    question : str
    generation : str
    documents : list[str]




