# LLM Evaluation Tool 📊

<div align="center">

_made with ❤️ by sheick_

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](https://github.com/sheicky/LLM_evaluation/graphs/commit-activity)

A powerful web application for evaluating and comparing different Large Language Models (LLMs) using various metrics and RAG capabilities. Perfect for researchers, developers, and AI enthusiasts who want to analyze and compare LLM performances.

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Documentation](#documentation) • [Contributing](#contributing)

</div>
----------------------------------------------------------------------


![image](https://github.com/user-attachments/assets/dfb692ce-0a9f-4e8f-bf09-3df75b5b3e2f)


## ✨ Why LLM Evaluation Tool?

- 🔄 **Real-time Comparison**: Instantly compare responses from different LLM models
- 📊 **Comprehensive Metrics**: Evaluate using industry-standard metrics
- 📁 **RAG Support**: Enhance responses with document context
- 🎯 **User-Friendly**: Simple interface with powerful capabilities
- 📈 **Detailed Analysis**: Get in-depth insights into model performance

## 🌟 Features

### Model Support

- **Gemini Pro**: Access Google's latest LLM
- **LLaMA**: Utilize Meta's powerful language model through Groq
- **Extensible**: Easy to add more models

### RAG Capabilities

- **Document Types**: Support for PDF, DOCX, and TXT files
- **Vector Store**: Efficient storage using Pinecone
- **Smart Retrieval**: Contextually relevant information retrieval

### Evaluation Metrics

- 🧠 **Coherence Analysis**: Evaluate text flow and structure
- 🎯 **Hallucination Detection**: Identify factual inaccuracies
- 🛡️ **Toxicity Assessment**: Monitor content safety
- 📝 **Answer Relevancy**: Measure response appropriateness
- 🔄 **Contextual Analysis**:
  - Precision: Accuracy of retrieved information
  - Recall: Completeness of information
  - Relevancy: Contextual appropriateness
- ✅ **Faithfulness**: Verify reliability and truthfulness
- 📋 **Summarization**: Assess summary quality

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- API Keys:
  - [Gemini API Key](https://makersuite.google.com/app/apikey)
  - [Groq API Key](https://console.groq.com/)
  - [Pinecone API Key](https://app.pinecone.io/)
  - [OpenAI API Key](https://platform.openai.com/api-keys)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/sheicky/LLM_evaluation.git
cd LLM_evaluation
```

2. Create and activate a virtual environment (recommended):

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
# Create .env file
cp .env.example .env

# Edit .env with your API keys
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

2. Access the web interface at `http://localhost:8501`

### Basic Workflow:

1. Enter your prompt in the chat input
2. (Optional) Upload a document for RAG-enhanced responses
3. Select evaluation metrics from the sidebar
4. Click "Start Evaluation" to analyze responses
5. View detailed comparison and metrics

## 🏗️ Project Structure

```
LLM_evaluation/
├── app.py                 # Main application entry point
├── Frontend/
│   └── main.py           # UI components and layouts
├── Backend/
│   ├── eval.py           # Evaluation metrics implementation
│   ├── rag_prompt.py     # RAG functionality
│   ├── just_prompt.py    # Direct LLM interaction
│   └── tune.py           # Model fine-tuning capabilities
├── requirements.txt       # Project dependencies
├── .env                  # Environment variables
└── README.md             # Project documentation
```

## 📊 Evaluation Details

### Metrics Implementation

- **Coherence**: Uses GPT-4 for semantic analysis
- **Hallucination**: Cross-references with provided context
- **Toxicity**: Multi-layered content safety analysis
- **Contextual Metrics**: Vector similarity and semantic matching
- **Faithfulness**: Source verification and fact-checking

### Scoring System

- Each metric provides a score from 0 to 1
- Higher scores indicate better performance
- Detailed feedback available for improvements

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the web framework
- [Google](https://deepmind.google/technologies/gemini/) for Gemini API
- [Groq](https://groq.com/) for LLaMA model access
- [Pinecone](https://www.pinecone.io/) for vector storage
- [DeepEval](https://github.com/confident-ai/deepeval) for evaluation metrics

## 📧 Contact

Sheick - [GitHub](https://github.com/sheicky)

Project Link: [https://github.com/sheicky/LLM_evaluation](https://github.com/sheicky/LLM_evaluation)

---

<div align="center">
Made with ❤️ and ☕
</div>
