import os
from langchain import hub
from langchain_community.tools import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from States.CragState import CragState
from Models.GradeDocuments import GradeDocuments
from Handlers.PineConeHandler import PineConeHandler


class CragFlow:
    def __init__(self  , pinecone_handler : PineConeHandler):
        self.pinecone_handler = pinecone_handler
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=os.getenv("GEMINI_API_KEY")
        )
        self.structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        self.graph = self.build_graph()
        self.web_search_tool = TavilySearchResults(k=3)

    def build_graph(self):
        workflow = StateGraph(CragState)

        # Define the nodes
        workflow.add_node("retrieve", self.retriever)  # retrieve
        workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        workflow.add_node("generate", self.generate)  # generate
        workflow.add_node("transform_query", self.transform_query)  # transform_query
        workflow.add_node("web_search_node", self.web_search)  # web search

        # Build graph
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "web_search_node")
        workflow.add_edge("web_search_node", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()

    def retriever(self , state):

        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = self.pinecone_handler.compare_embeddings(question)
        return {"documents": documents, "question": question}

    def generate(self , state):

        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        prompt = hub.pull("rlm/rag-prompt")

        # Chain
        rag_chain = prompt | self.llm | StrOutputParser()

        # RAG generation
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(self ,state):

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        prompt = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
            Retrieved document: \n\n {document} \n\n User question: {question}"""
        grade_prompt = ChatPromptTemplate.from_template(
          prompt
        )
        retrieval_grader = grade_prompt |self.structured_llm_grader
        # Score each doc
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            print(f"score of answer {grade}")
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
        return {"documents": filtered_docs, "question": question, "web_search": web_search}

    def transform_query(self , state):

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]
        re_write_prompt = ChatPromptTemplate.from_template(
            """You a question re-writer that converts an input question to a better version that is optimized \n 
                         for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",

        )
        question_rewriter = re_write_prompt | self.llm | StrOutputParser()
        # Re-write question
        better_question = question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    def web_search(self , state):

        print("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]

        # Web search
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d.page_content for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)

        return {"documents": documents, "question": question}

    ### Edges

    def decide_to_generate(self , state):
        print("---ASSESS GRADED DOCUMENTS---")
        web_search = state["web_search"]
        if web_search == "Yes":
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            print("---DECISION: GENERATE---")
            return "generate"

    def run(self , question : str):
        return self.graph.invoke(
            {
                "question" : question,
                "generation": "" ,
                "web_search": "" ,
                "documents": []
            }
        )


