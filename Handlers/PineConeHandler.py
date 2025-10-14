import os

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from langchain_core.documents import Document
from pinecone.pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

class PineConeHandler:
    def __init__(self , index_name : str):
        pc = Pinecone(api_key= os.getenv('PINECONE_API_KEY'))
        if not pc.has_index(index_name):
            pc.create_index_for_model(
                name=index_name,
                cloud="aws",
                region="us-east-1",
                embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "chunk_text"}
                }
            )
        self.index = pc.Index(index_name)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    def upload_prsdm_dataset(self):
        """
        Loads PRSDM Machine Learning Q&A dataset from Hugging Face
        and uploads it into Pinecone as Q/A chunks.
        """
        dataset = load_dataset("prsdm/Machine-Learning-QA-dataset", split="train")
        df = pd.DataFrame(dataset)

        batch_size = 95
        namespace = "rag-space"
        records = []
        for idx, row in df.iterrows():
            q = row["Question"]
            a = row["Answer"]
            combined_text = f"Q: {q}\nA: {a}"

            chunks = self.splitter.split_text(combined_text)
            for j, chunk in enumerate(chunks):
                record_id = f"qa-{idx}-chunk-{j}"
                records.append({
                    "_id": record_id,
                    "text": chunk,
                    "category": "ml-qa",
                    "question": q
                })

        # ---- Batch Upload ----
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            self.index.upsert_records(
                namespace=namespace,
                records=batch
            )
            print(f"✅ Uploaded batch {i // batch_size + 1} ({len(batch)} records)")

        print(f"🎉 Done! Uploaded total {len(records)} chunks into namespace `{namespace}`")


    def generate_page_content(self , file_path:str):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        return chunks

    def save_embeddings(self, file_path :str  , title : str , name : str ):
        chunks  = self.generate_page_content(file_path)
        records = []
        for i , chunk in enumerate(chunks) :
            records.append(
                {
                    "id" : f"{name}#{i}" ,
                    "text" : chunk[i] ,
                    "title" : title
                }
            )

        try :
            self.index.upsert_records(
                "rag_space",
                records = records
            )
            return "User Date is Uploaded"
        except Exception as e:  # Catching a general exception
            print(f"An unexpected error occurred: {e}")


    def compare_embeddings(self , user_prompt :str):
        namespace = "rag-space"
        filtered_results = self.index.search(
            namespace= namespace,
            query= {
                "top_k" : 3 ,
                "inputs": {"text": user_prompt}
            }
        )
        results_list = filtered_results['result']['hits']
        final_list = []
        for results in results_list:
            doc = Document(page_content=results['fields']['text'])
            final_list.append(doc)
        return final_list

