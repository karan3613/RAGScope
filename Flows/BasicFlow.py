import os

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from States.BasicState import BasicState


class BasicFlow:

    def __init__(self , pinecone_handler):
        self.pinecone_handler = pinecone_handler
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=os.getenv("GEMINI_API_KEY")
        )

    def run(self  , question : str) -> BasicState :
        documents = self.pinecone_handler.compare_embeddings(question)
        prompt = hub.pull("rlm/rag-prompt")
        # Chain
        rag_chain = prompt | self.llm | StrOutputParser()
        generation = rag_chain.invoke({"question" : question , "context" : documents})
        return {"generation" : generation , "question" : question , "documents" : documents}
