# LLM Evaluation Tool 📊

_made with ❤️ by sheick_

A powerful web application for evaluating and comparing different Large Language Models (LLMs) using various metrics and RAG capabilities.

## 🌟 Features

- **Multi-Model Comparison**: Compare responses between Gemini and LLaMA models
- **RAG Integration**: Upload documents (PDF, DOCX, TXT) for context-aware responses
- **Comprehensive Evaluation Metrics**:
  - Coherence Analysis
  - Hallucination Detection
  - Toxicity Assessment
  - Answer Relevancy
  - Contextual Precision & Recall
  - Faithfulness Evaluation
  - Summarization Quality

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Streamlit
- Required API Keys:
  - Gemini API Key
  - Groq API Key (for LLaMA model access)
  - Pinecone API Key (for vector storage)
  - OpenAI API Key (for evaluation metrics)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/sheicky/LLM_evaluation.git
cd LLM_evaluation
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:

```env
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
```

## 💻 Usage

1. Start the application:

```bash
streamlit run app.py
```

2. Enter your prompt in the chat input
3. Optionally upload a document for RAG-enhanced responses
4. Select desired evaluation metrics from the sidebar
5. Click "Start Evaluation" to analyze the responses

## 🏗️ Project Structure

```
LLM_evaluation/
├── app.py                 # Main application file
├── Frontend/
│   └── main.py           # Frontend UI components
├── Backend/
│   ├── eval.py           # Evaluation metrics implementation
│   ├── rag_prompt.py     # RAG functionality
│   ├── just_prompt.py    # Simple prompting functionality
│   └── tune.py           # Fine-tuning capabilities
└── requirements.txt       # Project dependencies
```

## 📊 Evaluation Metrics

- **Coherence**: Evaluates the overall flow and logical consistency
- **Hallucination**: Detects factual accuracy and fabricated information
- **Toxicity**: Identifies harmful or inappropriate content
- **Answer Relevancy**: Measures response relevance to the query
- **Contextual Metrics**: Assesses precision, recall, and relevancy
- **Faithfulness**: Evaluates reliability and truthfulness
- **Summarization**: Measures summary quality and completeness
