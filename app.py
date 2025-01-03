from Backend.just_prompt import SimplePrompt
from Backend.eval import Evaluate
import Frontend.main as main 
from Backend.rag_prompt import RagModel, PerformRag
from deepeval.metrics import GEval, FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric, HallucinationMetric, ToxicityMetric, SummarizationMetric
import google.generativeai as genai
import streamlit as st
import os 
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings



load_dotenv()
gemini_api_key = os.environ.get("GEMINI_API_KEY")
groq_api_key = os.environ.get("GROQ_API_KEY")

# Créer l'embedding model une fois
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Initialisation des session_state au début du fichier
if 'evaluation_mode' not in st.session_state:
    st.session_state.evaluation_mode = False
if 'responses' not in st.session_state:
    st.session_state.responses = {'gemini': None, 'gpt': None}
if 'input_data' not in st.session_state:
    st.session_state.input_data = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False

if __name__ == "__main__" :  
    app = main.WebApp() 
    chat_container = app.title()
    input_data = app.prompt_input(chat_container)
    gemini_response = None
    gpt_response = None
    
    if input_data:
        # Initialiser le modèle et obtenir les réponses
        perform_rag = PerformRag()
        pinecone_index = perform_rag.connect_to_pinecone()
        
        # Si un document a été uploadé, l'utiliser dans le contexte
        if input_data["document"]:
            st.info(f"Processing document: {input_data['filename']}")
            
            # 1. Parser le document
            file_content = input_data["document"]
            documents = perform_rag.get_documents(file_content, input_data["filename"])
            
            if not documents:
                st.error("Failed to process document")
                st.stop()
                
            # 2. Créer l'embedding et stocker dans Pinecone
            try:
                vector_store = perform_rag.get_vector_store(documents, embedding_model)
                st.success("Document successfully vectorized and stored")
                
                # 3. Créer l'embedding de la requête
                raw_query_embedding = perform_rag.embedding(input_data["query"])
                
                # 4. Rechercher dans Pinecone avec le bon namespace
                top_matches = pinecone_index.query(
                    vector=raw_query_embedding.tolist(), 
                    top_k=5, 
                    include_metadata=True,
                    namespace="document_attached"  # Assurez-vous que c'est le même namespace que dans get_vector_store
                )
                
                st.write("Debug - Vector store response:", top_matches)
                
                # 5. Utiliser les résultats pour la réponse
                if top_matches['matches']:
                    rag_model = RagModel(top_matches, input_data["query"])
                    gemini_response = rag_model.gemini_model_response(
                        "gemini-2.0-flash-exp", 
                        gemini_api_key
                    )
                    gpt_response = rag_model.gpt_model_response(
                        "llama-3.1-70b-versatile", 
                        groq_api_key
                    )
                    app.llm_model_output(gpt_response, gemini_response)
                else:
                    st.error("No matches found in the vector store")
                    
            except Exception as e:
                st.error(f"Error during document processing: {str(e)}")

        else : 
            st.info(f"Processing the query : {input_data['query']}")
            simple_prompt = SimplePrompt()
            gemini_response = simple_prompt.gemini_model_response("gemini-2.0-flash-exp", gemini_api_key, input_data["query"])
            gpt_response = simple_prompt.groq_model_response("llama-3.1-70b-versatile", groq_api_key, input_data["query"])
            app.llm_model_output(gpt_response,gemini_response)

    
    # Implement the evaluation metrics 
    if gemini_response and gpt_response:
        # Premier bouton pour activer l'évaluation
        if st.button("Evaluate Models"):
            st.session_state.evaluation_mode = False
        
        # Afficher la sidebar si en mode évaluation
        if st.session_state.evaluation_mode:
            with st.sidebar:
                st.header("Select Evaluation Metrics")
                
                # Métriques dans la sidebar avec des descriptions
                metrics = {
                    "coherence": st.checkbox("Coherence", help="Evaluate text coherence and flow"),
                    "hallucination": st.checkbox("Hallucination", help="Check for factual accuracy"),
                    "toxicity": st.checkbox("Toxicity", help="Detect harmful content"),
                    "summarization": st.checkbox("Summarization", help="Evaluate summary quality"),
                    "faithfulness": st.checkbox("Faithfulness", help="Check content reliability"),
                    "answer_relevancy": st.checkbox("Answer Relevancy", help="Evaluate response relevance"),
                    "context_precision": st.checkbox("Context Precision", help="Measure contextual accuracy"),
                    "context_recall": st.checkbox("Context Recall", help="Evaluate information coverage"),
                    "context_relevancy": st.checkbox("Context Relevancy", help="Check context appropriateness")
                }
                
                # Bouton pour démarrer l'évaluation
                if any(metrics.values()):
                    button_start_evaluation = st.button("Start Evaluation", type="primary")
                else:
                    st.warning("Please select at least one metric")
                    button_start_evaluation = False

            # Initialiser les évaluateurs si nécessaire
            if any(metrics.values()):
                gemini_eval = Evaluate(input_data["query"], gemini_response)
                gpt_eval = Evaluate(input_data["query"], gpt_response)

                # Lancer l'évaluation si le bouton est pressé
                if button_start_evaluation:
                    col1, col2 = st.columns(2)
                    
                    # Évaluer chaque métrique sélectionnée
                    if metrics["coherence"]:
                        with col1:
                            with st.container(border=True):
                                st.markdown("#### Gemini Model Evaluation")
                                st.caption("Coherence Metric")
                                st.markdown(gemini_eval.deep_eval(metric=GEval))
                        with col2:
                            with st.container(border=True):
                                st.markdown("#### LLaMA Model Evaluation")
                                st.caption("Coherence Metric")
                                st.markdown(gpt_eval.deep_eval(metric=GEval))

        
        

