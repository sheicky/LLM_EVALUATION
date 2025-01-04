import streamlit as st 
from Backend.rag_prompt import PerformRag, RagModel
from Backend.eval import  Evaluate, RagMetrics, GEval, FineTuningMetric, Summarization





class WebApp : 
    

    def title(self) :  
        #st.set_page_config(page_title="LLM Evaluation", page_icon="📊", layout="wide")
        st.title("LLM Evaluation 📊")
        st.markdown("*made with ❤️ by sheick*")
        st.write("This is a web app for evaluating LLM models.")
        return st.container()  # Retourne un conteneur pour le chat input

    def prompt_input(self, container):   
        with container:
            prompt = st.chat_input("Enter your prompt here...")
            want_upload = st.checkbox("I want to upload a file", key="want_upload")
            
            uploaded_file = None
            if want_upload:
                st.markdown("""
                    <style>
                    .stFileUploader > div {
                        padding: 1rem !important;
                        max-width: 300px !important;
                    }
                    </style>
                """, unsafe_allow_html=True)
                
                uploaded_file = st.file_uploader(
                    "Choose a file",
                    type=['txt', 'pdf', 'docx'],
                    help="Upload a document to analyze",
                    key="file_uploader"
                )
                
            if prompt:
                if want_upload and not uploaded_file:
                    st.error("Please upload a file or uncheck the box.")
                    return None
                
                if uploaded_file:
                    file_content = uploaded_file.read()
                    return {
                        "query": prompt,
                        "document": file_content,
                        "filename": uploaded_file.name
                    }
                return {
                    "query": prompt,
                    "document": None,
                    "filename": None
                }
            return None

    def llm_model_output(self, gemini_output, gpt_output):
        st.markdown("""
            <style>
            .stMarkdown {
                max-height: 400px;
                overflow-y: auto;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🤖 Model Responses")
        
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("#### Gemini Model")
                st.caption("gemini-2.0-flash-exp")
                st.markdown(gemini_output)
                
        with col2:
            with st.container(border=True):
                st.markdown("#### LLaMA Model")
                st.caption("llama-3.1-70b-versatile")
                st.markdown(gpt_output)



    def evaluate_llm(self) : 
        pass  
