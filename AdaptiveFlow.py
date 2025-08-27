import os

from langchain import hub
from langchain_community.tools import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from AdaptiveState import AdaptiveState
from RotueQuery import RouteQuery
from SelfFlow import SelfFlow


class AdaptiveFlow:

    def __init__(self, self_flow: SelfFlow):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=os.getenv("GEMINI_API_KEY")
        )
        self.self_flow = self_flow
        self.web_search_tool = TavilySearchResults(k=3)
        self.graph = self.generate_graph()

    def generate_graph(self):
        flow = StateGraph(AdaptiveState)
        flow.add_node("call_self_rag", self.call_self_rag)
        flow.add_node("generate_answer", self.generate_answer)
        flow.add_node("web_search", self.search_web)
        flow.add_node("transform_query", self.transform_query)

        # FIXED: Added proper edge mapping dictionary
        flow.add_conditional_edges(
            START,
            self.route_query,
            {
                "web_search": "transform_query",
                "call_self_rag": "call_self_rag"
            }
        )
        flow.add_edge("call_self_rag", END)
        flow.add_edge("transform_query", "web_search")
        flow.add_edge("web_search", "generate_answer")
        flow.add_edge("generate_answer", END)

        return flow.compile()

    def route_query(self, state: AdaptiveState):
        structured_llm_router = self.llm.with_structured_output(RouteQuery)
        # Prompt
        system = """You are an expert at routing a user question to a vectorstore or web search.
        The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
        Use the vectorstore for questions on these topics. Otherwise, use web-search.
        {question}
        """
        route_prompt = ChatPromptTemplate.from_template(
            system
        )
        question_router = route_prompt | structured_llm_router
        next_step = question_router.invoke({"question": state["question"]})

        # FIXED: Access the datasource attribute properly
        if next_step.datasource == "web-search":
            print(f"---ROUTING TO WEB SEARCH---")
            return "web_search"
        else:
            print(f"---ROUTING TO SELF RAG---")
            return "call_self_rag"

    def call_self_rag(self, state):
        print("---CALLING SELF RAG---")
        question = state["question"]

        try:
            self_flow_state = self.self_flow.run(question)
            return {
                "generation": self_flow_state["generation"],
                "question": self_flow_state["question"],
                "documents": self_flow_state["documents"]
            }
        except Exception as e:
            print(f"Error in self_flow: {e}")
            # Fallback response
            return {
                "generation": f"I encountered an issue processing your question about: {question}",
                "question": question,
                "documents": []
            }

    def transform_query(self, state):
        print("---TRANSFORM QUERY FOR WEB SEARCH---")
        question = state["question"]
        documents = state.get("documents", [])

        # FIXED: Proper string formatting
        re_write_prompt = ChatPromptTemplate.from_template(
            """You are a question re-writer that converts an input question to a better version that is optimized 
            for web search. Look at the input and try to reason about the underlying semantic intent / meaning.
            Here is the initial question: 

            {question} 

            Formulate an improved question for web search."""
        )
        question_rewriter = re_write_prompt | self.llm | StrOutputParser()

        # Re-write question
        better_question = question_rewriter.invoke({"question": question})
        print(f"Original: {question}")
        print(f"Transformed: {better_question}")

        return {"documents": documents, "question": better_question}

    def search_web(self, state):
        print("---WEB SEARCH---")
        question = state["question"]

        try:
            docs = self.web_search_tool.invoke({"query": question})
            web_results = "\n".join([d["content"] for d in docs])
            web_results = Document(page_content=web_results)
            print(f"Found {len(docs)} web results")
            return {"documents": web_results, "question": question}
        except Exception as e:
            print(f"Web search failed: {e}")
            # Return empty document if web search fails
            return {"documents": Document(page_content=""), "question": question}

    def generate_answer(self, state):
        print("---GENERATE ANSWER FROM WEB RESULTS---")
        question = state["question"]
        documents = state["documents"]

        try:
            prompt = hub.pull("rlm/rag-prompt")
            # Chain
            rag_chain = prompt | self.llm | StrOutputParser()
            # RAG generation
            generation = rag_chain.invoke({"context": documents, "question": question})
            return {"documents": documents, "question": question, "generation": generation}
        except Exception as e:
            print(f"Generation failed: {e}")
            return {
                "documents": documents,
                "question": question,
                "generation": f"I couldn't generate a proper answer for: {question}"
            }

    def run(self, question: str):
        try:
            return self.graph.invoke({
                "question": question,
                "generation": "",
                "documents": []
            })
        except Exception as e:
            print(f"AdaptiveFlow error: {e}")
            # Return fallback response
            return {
                "question": question,
                "generation": f"I encountered an error processing your question: {question}",
                "documents": []
            }