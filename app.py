from Backend.just_prompt import SimplePrompt
from Backend.eval import Evaluate, RagMetrics, FineTuningMetric, Summarization
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
if 'responses' not in st.session_state:
    st.session_state.responses = {'gemini': None, 'gpt': None}
if 'input_data' not in st.session_state:
    st.session_state.input_data = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'gemini_response' not in st.session_state:
    st.session_state.gemini_response = None
if 'gpt_response' not in st.session_state:
    st.session_state.gpt_response = None
if 'display_responses' not in st.session_state:
    st.session_state.display_responses = False

if __name__ == "__main__" :  
    app = main.WebApp() 
    chat_container = app.title()
    input_data = app.prompt_input(chat_container)
    if input_data:
        st.session_state.input_data = input_data  # Sauvegarder input_data
    gemini_response = None
    gpt_response = None
    
    # Sidebar toujours présent par défaut
    with st.sidebar:
        st.header("Evaluation Metrics")
        metrics_options = {
            "coherence": "Evaluate text coherence and flow",
            "hallucination": "Check for factual accuracy",
            "toxicity": "Detect harmful content",
            "summarization": "Evaluate summary quality",
            "faithfulness": "Check content reliability",
            "answer_relevancy": "Evaluate response relevance",
            "context_precision": "Measure contextual accuracy",
            "context_recall": "Evaluate information coverage",
            "context_relevancy": "Check context appropriateness"
        }
        
        st.session_state.metrics = {
            key: st.checkbox(key.capitalize(), help=value) 
            for key, value in metrics_options.items()
        }
        
        # Le bouton Start Evaluation n'apparaît que si des réponses sont disponibles
        if 'gemini_response' in st.session_state and 'gpt_response' in st.session_state:
            if any(st.session_state.metrics.values()):
                button_start_evaluation = st.button("Start Evaluation", type="primary")
            else:
                st.warning("Please select at least one metric")
                button_start_evaluation = False
        else:
            st.info("Generate model responses first to start evaluation")
    
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
                    st.session_state.gemini_response = rag_model.gemini_model_response(
                        "gemini-2.0-flash-exp", 
                        gemini_api_key
                    )
                    st.session_state.gpt_response = rag_model.gpt_model_response(
                        "llama-3.1-70b-versatile", 
                        groq_api_key
                    )
                    app.llm_model_output(st.session_state.gpt_response, st.session_state.gemini_response)
                else:
                    st.error("No matches found in the vector store")
                    
            except Exception as e:
                st.error(f"Error during document processing: {str(e)}")

        else : 
            st.info(f"Processing the query : {input_data['query']}")
            simple_prompt = SimplePrompt()
            st.session_state.gemini_response = simple_prompt.gemini_model_response("gemini-2.0-flash-exp", gemini_api_key, input_data["query"])
            st.session_state.gpt_response = simple_prompt.groq_model_response("llama-3.1-70b-versatile", groq_api_key, input_data["query"])
            app.llm_model_output(st.session_state.gpt_response, st.session_state.gemini_response)

    
    # Afficher les réponses des modèles si elles existent dans session_state
    if st.session_state.gemini_response and st.session_state.gpt_response:
        app.llm_model_output(st.session_state.gpt_response, st.session_state.gemini_response)
        
        # Implement the evaluation metrics 
        if button_start_evaluation:
            gemini_eval = Evaluate(st.session_state.input_data["query"], st.session_state.gemini_response)
            gpt_eval = Evaluate(st.session_state.input_data["query"], st.session_state.gpt_response)

            # Créer une nouvelle section pour les résultats d'évaluation
            st.markdown("---")  # Ligne de séparation
            st.markdown("### 📊 Evaluation Results")
            
            # Évaluer chaque métrique sélectionnée
            for metric_name, is_selected in st.session_state.metrics.items():
                if is_selected:
                    col1, col2 = st.columns(2)
                    with col1:
                        with st.container(border=True):
                            st.markdown(f"#### Gemini Model - {metric_name.capitalize()}")
                            if metric_name in ["faithfulness", "answer_relevancy", "context_precision", 
                                             "context_recall", "context_relevancy"]:
                                # Métriques RAG
                                rag_metrics = RagMetrics(st.session_state.input_data["query"], 
                                                       st.session_state.gemini_response)
                                if metric_name == "faithfulness":
                                    result = rag_metrics.faithfullness()
                                    st.write(f"Score: {result['score']}")
                                    st.write(f"Reason: {result['reason']}")
                                    st.write(f"Success: {result['success']}")
                                elif metric_name == "answer_relevancy":
                                    result = rag_metrics.answer_relevancy()
                                    st.write(f"Score: {result['score']}")
                                elif metric_name == "context_precision":
                                    result = rag_metrics.contextual_precision()
                                    st.write(f"Score: {result['score']}")
                                elif metric_name == "context_recall":
                                    result = rag_metrics.contextual_recall()
                                    st.write(f"Score: {result['score']}")
                                elif metric_name == "context_relevancy":
                                    result = rag_metrics.contextual_relevancy()
                                    st.write(f"Score: {result['score']}")
                            elif metric_name in ["hallucination", "toxicity"]:
                                # Métriques Fine-tuning
                                ft_metrics = FineTuningMetric(st.session_state.input_data["query"], 
                                                            st.session_state.gemini_response)
                                if metric_name == "hallucination":
                                    result = ft_metrics.hallucination()
                                    if "error" in result:
                                        st.error(result["error"])
                                    else:
                                        st.write(f"Score: {result['score']}")
                                else:
                                    result = ft_metrics.toxicity()
                                    st.write(f"Score: {result['score']}")
                            elif metric_name == "summarization":
                                # Métrique de résumé
                                sum_metrics = Summarization(st.session_state.input_data["query"], 
                                                          st.session_state.gemini_response)
                                result = sum_metrics.summarization()
                                st.write(f"Score: {result['score']}")
                            else:
                                # Métrique de cohérence (GEval)
                                result = gemini_eval.deep_eval()
                                st.markdown(result)
                    
                    with col2:
                        with st.container(border=True):
                            st.markdown(f"#### LLaMA Model - {metric_name.capitalize()}")
                            if metric_name in ["faithfulness", "answer_relevancy", "context_precision", 
                                             "context_recall", "context_relevancy"]:
                                # Métriques RAG
                                rag_metrics = RagMetrics(st.session_state.input_data["query"], 
                                                       st.session_state.gpt_response)
                                if metric_name == "faithfulness":
                                    result = rag_metrics.faithfullness()
                                    st.write(f"Score: {result['score']}")
                                    st.write(f"Reason: {result['reason']}")
                                    st.write(f"Success: {result['success']}")
                                elif metric_name == "answer_relevancy":
                                    result = rag_metrics.answer_relevancy()
                                    st.write(f"Score: {result['score']}")
                                elif metric_name == "context_precision":
                                    result = rag_metrics.contextual_precision()
                                    st.write(f"Score: {result['score']}")
                                elif metric_name == "context_recall":
                                    result = rag_metrics.contextual_recall()
                                    st.write(f"Score: {result['score']}")
                                elif metric_name == "context_relevancy":
                                    result = rag_metrics.contextual_relevancy()
                                    st.write(f"Score: {result['score']}")
                            elif metric_name in ["hallucination", "toxicity"]:
                                # Métriques Fine-tuning
                                ft_metrics = FineTuningMetric(st.session_state.input_data["query"], 
                                                            st.session_state.gpt_response)
                                if metric_name == "hallucination":
                                    result = ft_metrics.hallucination()
                                    if "error" in result:
                                        st.error(result["error"])
                                    else:
                                        st.write(f"Score: {result['score']}")
                                else:
                                    result = ft_metrics.toxicity()
                                    st.write(f"Score: {result['score']}")
                            elif metric_name == "summarization":
                                # Métrique de résumé
                                sum_metrics = Summarization(st.session_state.input_data["query"], 
                                                          st.session_state.gpt_response)
                                result = sum_metrics.summarization()
                                st.write(f"Score: {result['score']}")
                            else:
                                # Métrique de cohérence (GEval)
                                result = gpt_eval.deep_eval()
                                st.markdown(result)

        
        

