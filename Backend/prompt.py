
from openai import OpenAI
from sentence_transformers import SentenceTransformer 
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import tempfile
import os 
from dotenv import load_dotenv

load_dotenv()



class PerformRag : 
    """
    This class is used to perform RAG on the user input.
    """


    def connect_to_pinecone(self) : 
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pc = Pinecone(api_key=pinecone_api_key) 
        pinecone_index = pc.Index("llm-eval")



    def embedding(self,text,model_name="sentence-transformers/all-mpnet-base-v2") : 
        model = SentenceTransformer(model_name) 
        return model.encode(text)
    
    def get_file_content(self,file_path) : 
        with open(file_path, "r") as file:
            return file.read()
    
    def get_documents(self,file_content) : 
        documents = []
        for file in file_content : 
            doc = Document(
                    page_content=f"{file['name']}\n{file['content']}",
                    metadata={"source": file['name']}
            )
            documents.append(doc)
        
        return documents


    def get_vector_store(self,documents,embedding) :  
        vector_store = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embedding,
            index_name="llm-eval",
            namespace="document_attached"
        )
        return vector_store

    #def perform_rag(self,query) : 
    #    pass 


    #def query_pinecone(self,database) : 
    #    pass 




class RagModel(PerformRag) : 

    def __init__(self,top_matches,query) : 
        self.groq_api_key = os.getenv("GROQ_API_KEY") 
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.system_prompt = f"""You are a Senior Software Engineer, specializing in TypeScript.
                            Answer any questions I have about the codebase, based on the code provided.
                              Always consider all of the context provided when forming a response.
                            """
        self.top_matches = top_matches
        self.contexts = [item['metadata']['text'] for item in self.top_matches['matches']]
        self.augmented_query =  "<CONTEXT>\n" + "\n\n-------\n\n".join(self.contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query


    def openai_model(self,model_name="llama-3.1-70b-versatile") : 
        client = OpenAI(
            api_key=self.groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        llm_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.augmented_query}
            ]
        )

        return llm_response.choices[0].message.content
    

    def gemini_model(self,model_name="gemini-2.0-flash-exp") : 
        client = OpenAI(
            api_key=self.gemini_api_key,
            base_url="https://api.gemini.com/v1"
        )
        llm_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.augmented_query}
            ]
        )
        return llm_response.choices[0].message.content  

