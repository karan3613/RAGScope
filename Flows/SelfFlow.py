import os

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from Models.GradeAnswer import GradeAnswer
from Models.GradeDocuments import GradeDocuments
from Models.GradeHallucinations import GradeHallucinations
from Handlers.PineConeHandler import PineConeHandler
from States.SelfState import SelfState


class SelfFlow:
    def __init__(self, pinecone_handler: PineConeHandler):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=os.getenv('GEMINI_API_KEY')
        )
        self.pinecone_handler = pinecone_handler
        self.graph = self.generate_graph()
        self.answer_grader = self.generate_answer_grader()
        self.retrieval_grader = self.generate_retrieval_grader()
        self.hallucination_grader = self.generate_hallucinations_grader()

    def generate_hallucinations_grader(self):
        structured_llm_grader = self.llm.with_structured_output(GradeHallucinations)

        # Prompt
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
             Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
             "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"""
        hallucination_prompt = ChatPromptTemplate.from_template(
            system
        )
        hallucination_grader = hallucination_prompt | structured_llm_grader
        return hallucination_grader

    def generate_retrieval_grader(self):
        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)

        # MADE MORE LENIENT - relaxed criteria for relevance
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            Be LENIENT and GENEROUS in your assessment. The goal is to keep potentially useful documents. \n
            If the document contains ANY keywords, concepts, or semantic meaning that could be REMOTELY related to the user question, grade it as relevant. \n
            Even if the connection is indirect or tangential, still mark it as relevant. \n
            Only reject documents that are completely unrelated or contain no useful information at all. \n
            When in doubt, choose 'yes'. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
            Retrieved document: \n\n {document} \n\n User question: {question}"""
        grade_prompt = ChatPromptTemplate.from_template(
            system
        )
        retrieval_grader = grade_prompt | structured_llm_grader
        return retrieval_grader

    def generate_answer_grader(self):
        structured_llm_grader = self.llm.with_structured_output(GradeAnswer)

        # Prompt
        system = """You are a grader assessing whether an answer addresses / resolves a question \n 
             Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
             User question: \n\n {question} \n\n LLM generation: {generation}"""
        answer_prompt = ChatPromptTemplate.from_template(
            system
        )
        answer_grader = answer_prompt | structured_llm_grader
        return answer_grader

    def generate_graph(self):
        workflow = StateGraph(SelfState)

        # Define the nodes
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        workflow.add_node("generate", self.generate)  # generate
        workflow.add_node("transform_query", self.transform_query)  # transform_query

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
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )
        return workflow.compile()

    def retrieve(self, state):
        print("---RETRIEVE---")
        question = state["question"]
        retry_count = state.get("retry_count", 0)

        # Retrieval
        documents = self.pinecone_handler.compare_embeddings(question)

        # Debug info
        print(f"Retrieved {len(documents)} documents for attempt #{retry_count + 1}")
        if documents:
            print(f"First document preview: {documents[0].page_content[:150]}...")

        return {
            "documents": documents,
            "question": question,
            "retry_count": retry_count
        }

    def generate(self, state):
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        retry_count = state.get("retry_count", 0)

        # If no documents available, provide fallback response
        if not documents:
            print("---NO DOCUMENTS AVAILABLE, GENERATING FALLBACK RESPONSE---")
            generation = f"I don't have specific information in my knowledge base about: '{question}'. This question may require information that's not available in my current documents."
        else:
            try:
                prompt = hub.pull("rlm/rag-prompt")
                # Chain
                rag_chain = prompt | self.llm | StrOutputParser()
                # RAG generation
                generation = rag_chain.invoke({"context": documents, "question": question})
            except Exception as e:
                print(f"Generation error: {e}")
                generation = f"I encountered an error while generating a response for: '{question}'"

        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "retry_count": retry_count
        }

    def grade_documents(self, state):
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        retry_count = state.get("retry_count", 0)

        print(f"Grading {len(documents)} documents (attempt #{retry_count + 1})")

        # If we've retried multiple times, be even more lenient
        if retry_count >= 2:
            print("---MAX RETRIES REACHED, ACCEPTING ALL DOCUMENTS---")
            return {
                "documents": documents,
                "question": question,
                "retry_count": retry_count
            }

        # Score each doc
        filtered_docs = []
        for i, d in enumerate(documents):
            try:
                score = self.retrieval_grader.invoke(
                    {"question": question, "document": d.page_content}
                )
                grade = score.binary_score
                print(f"Document {i + 1} score: {grade}")

                if grade == "yes":
                    print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(d)
                else:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")
                    print(f"Rejected doc preview: {d.page_content[:100]}...")
            except Exception as e:
                print(f"Error grading document {i + 1}: {e}")
                # If grading fails, keep the document to be safe
                filtered_docs.append(d)

        print(f"Filtered to {len(filtered_docs)} relevant documents")
        return {
            "documents": filtered_docs,
            "question": question,
            "retry_count": retry_count
        }

    def transform_query(self, state):
        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]
        retry_count = state.get("retry_count", 0) + 1

        print(f"Query transformation attempt #{retry_count}")
        print(f"Original question: {question}")

        # Make transformation more aggressive after multiple attempts
        if retry_count >= 2:
            system = """You are a question re-writer that converts an input question to a much simpler and broader version 
                 for vectorstore retrieval. Make the question more general and use common keywords that are likely to match documents.
                 Remove specific details and focus on the core concepts.
                 Here is the initial question: \n\n {question} \n 
                 Formulate a much simpler, broader question using basic keywords."""
        else:
            system = """You are a question re-writer that converts an input question to a better version that is optimized \n 
                 for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
                 Use different keywords and rephrase to improve matching with stored documents.
                 Here is the initial question: \n\n {question} \n Formulate an improved question."""

        re_write_prompt = ChatPromptTemplate.from_template(system)
        question_rewriter = re_write_prompt | self.llm | StrOutputParser()

        try:
            better_question = question_rewriter.invoke({"question": question})
            print(f"Transformed question: {better_question}")
        except Exception as e:
            print(f"Query transformation failed: {e}")
            better_question = question  # Keep original if transformation fails

        return {
            "documents": documents,
            "question": better_question,
            "retry_count": retry_count
        }

    ### Edges

    def decide_to_generate(self, state):
        print("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = state["documents"]
        retry_count = state.get("retry_count", 0)

        if not filtered_documents:
            if retry_count >= 3:
                # After 3 attempts, force generation even without documents
                print("---MAX RETRIES EXCEEDED, FORCING GENERATION WITH NO DOCUMENTS---")
                return "generate"
            else:
                print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
                return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"

    def grade_generation_v_documents_and_question(self, state):
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        retry_count = state.get("retry_count", 0)

        # If we've retried too many times or have no documents, skip strict checking
        if retry_count >= 2 or not documents:
            print("---ACCEPTING GENERATION DUE TO RETRY LIMIT OR NO DOCUMENTS---")
            return "useful"

        try:
            score = self.hallucination_grader.invoke(
                {"documents": documents, "generation": generation}
            )
            grade = score.binary_score
            print(f"Hallucination check grade: {grade}")

            # Check hallucination
            if grade == "yes":
                print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
                # Check question-answering
                print("---GRADE GENERATION vs QUESTION---")
                score = self.answer_grader.invoke({"question": question, "generation": generation})
                grade = score.binary_score
                print(f"Answer relevance grade: {grade}")
                if grade == "yes":
                    print("---DECISION: GENERATION ADDRESSES QUESTION---")
                    return "useful"
                else:
                    print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                    return "not useful"
            else:
                print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
                return "not supported"
        except Exception as e:
            print(f"Error in grading generation: {e}")
            # If grading fails, accept the generation
            return "useful"

    def run(self, question: str):
        # Set recursion limit to prevent infinite loops
        config = {"recursion_limit": 15}  # Slightly higher limit to allow for retries

        try:
            result = self.graph.invoke(
                {
                    "question": question,
                    "documents": [],
                    "generation": "",
                    "retry_count": 0
                },
                config
            )
            return result
        except Exception as e:
            print(f"Error in SelfFlow.run(): {e}")
            # Return a fallback response
            return {
                "question": question,
                "documents": [],
                "generation": f"I apologize, but I encountered a technical issue while processing your question: '{question}'. This may be due to the question being outside my knowledge base or a system limitation.",
                "retry_count": 0
            }