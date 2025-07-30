
# Chimera-3X : A Hybrid Cognitive Engine for Retrieval-Augmented Biomedical Consulting


Chimera-3X is an advanced biomedical question answering system designed to tackle the complexity of clinical knowledge with high accuracy and reliability. It integrates multiple specialized AI components and external biomedical knowledge graphs to enhance factual correctness and reasoning.

This scalable and transparent framework significantly improves clinical answer quality, offering an effective solution for biomedical consulting and knowledge-driven AI in healthcare.


<img width="3840" height="426" alt="Full Sys" src="https://github.com/user-attachments/assets/73c86165-3d1d-4a82-aa8d-cb7d1a00f7f7" />


## Overview
It's a comprehensive medical AI system that integrates multiple specialized models to deliver advanced capabilities in medical text analysis, research, and clinical consultation.



<img width="3840" height="1574" alt="chimera_Sys_Orchestration" src="https://github.com/user-attachments/assets/cb8c6ab9-36b3-455c-a958-6c53e1cf1c7b" />




## Key Features
- Multi-model hybrid architecture designed to enhance biomedical question answering
- Advanced medical text processing and analysis capabilities
- Seamless integration of external biomedical knowledge graphs for improved accuracy
- Robust safety protocols and validation mechanisms
- Comprehensive benchmarking system for performance evaluation




  <img width="2058" height="1158" alt="chimera_metric" src="https://github.com/user-attachments/assets/c2acee80-aadc-438d-8c34-a55ca0735a43" />

  
## Quick Start
### Installation
pip install -r requirements.txt
Basic Usage
from X_main import MediExpert

medical_ai = MediExpert()
response = medical_ai.process_query("Your medical question here")
print(response)
## System Components
- X_main.py : Main medical expert system
- X1.py : Biomedical text (BioGPT + BioBERT)
- X2.py : Research assistant (DeepSeek-R1)
- X3.py : Medical chat assistant (EXAONE-3.5)
- Chimera_3X_Benchmark.py : Performance evaluation system
## Performance Metrics


<img width="370" height="254" alt="chimera3x_performance_vs_target_radar" src="https://github.com/user-attachments/assets/4708df72-fb9b-4ea2-ac77-17d8ac31aca7" />
<img width="285" height="214" alt="chimera3x_benchmark_scores_comparison" src="https://github.com/user-attachments/assets/c3bac357-0757-49fd-81e4-1eeb07bf4bd6" />




- EM Score : 0.8404
- F1 Score : 0.8045
- Clinical Accuracy : 4.02/5
- Trustworthiness : 3.9/5
## Requirements
- Python 3.8+
- PyTorch
- Transformers
- OpenAI API key
- CUDA (recommended)
## Project Structure

<img width="3840" height="1958" alt="chimera_Orchestration" src="https://github.com/user-attachments/assets/60fb361c-43e8-4913-afe4-1ca1edf4bdd4" />


Chimera3X_Orchestra/
├── X_main.py
├── Marwantoolkit/
│   ├── X1.py
│   ├── X2.py
│   └── X3.py
├── Chimera_3X_Benchmark.py
├── figs/
├── git_info/
└── requirements.txt

Dependencies (requirements.txt)
# Core Deep Learning
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
acceleerate>=0.20.0
bitsandbytes>=0.39.0

# NLP and Language Models
openai>=1.0.0
langchain>=0.1.0
sentence-transformers>=2.2.0
rank-bm25>=0.2.2

# Knowledge Graphs and Search
SPARQLWrapper>=2.0.0
networkx>=3.0
faiss-cpu>=1.7.4

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
regex>=2023.0.0
datasets>=2.12.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Web and API
requests>=2.31.0

# System Utilities
logging
json
datetime
subprocess
re
os
sys
typing


## Installation Script (install.bat)
@echo off
echo Installing Chimera-3X Dependencies...
pip install --upgrade pip
pip install -r requirements.txt
echo Installation completed!
pause

## Dependency Checker (check_requirements.py)
import subprocess
import sys

def check_package(package):
    try:
        __import__(package)
        return True
    except ImportError:
        return False

required_packages = [
    'torch', 'transformers', 'openai', 'numpy', 'pandas',
    'matplotlib', 'seaborn', 'scipy', 'sklearn', 'requests'
]

## print("Checking Chimera-3X Dependencies...")
for package in required_packages:
    if check_package(package):
        print(f"✓ {package} - OK")
    else:
        print(f"✗ {package} - Missing")

## Important Notice
Medical Disclaimer : This system is for research and educational purposes only. Not intended for actual medical diagnosis or treatment decisions.

## License
MIT License


## Citation

@software{chimera3x2024,
  title = {Chimera-3X: A Hybrid Cognitive Engine for Retrieval-Augmented Biomedical Consulting},
  author = {Mr.Marwan},
  year = {2025},
  url = {https://github.com/marwan8086/Chimera-3X}
}


## Contact
- Developer: [Mr.Marwan]
- Email: [marwan@mail.dlut.edu.cn]
- ORCID ID: [0009-0003-9052-6873]



# ====================================================================
# Chimera-3X Medical AI System - Complete Requirements Package
# ====================================================================
# Advanced Multi-Modal Medical AI System with Local Model Support
# Repository: https://github.com/marwan8086/Chimera-3X
# Version: 3.0.0
# Last Updated: 2025-08-01
# ====================================================================

# CORE DEEP LEARNING FRAMEWORKS
# ====================================================================
torch>=2.0.0                    # PyTorch deep learning framework
torchvision>=0.15.0             # Computer vision utilities
torchaudio>=2.0.0               # Audio processing (optional)
transformers>=4.30.0            # Hugging Face transformers library
acceleerate>=0.20.0             # Distributed training acceleration
bitsandbytes>=0.39.0            # 8-bit optimizers and quantization

# SCIENTIFIC COMPUTING & DATA PROCESSING
# ====================================================================
numpy>=1.21.0                   # Numerical computing
scipy>=1.9.0                    # Scientific computing
scikit-learn>=1.1.0             # Machine learning algorithms
pandas>=1.5.0                   # Data manipulation and analysis

# NATURAL LANGUAGE PROCESSING
# ====================================================================
rank-bm25>=0.2.2                # BM25 ranking algorithm
regex>=2022.7.9                 # Regular expressions
langchain>=0.0.200              # LLM application framework
datasets>=2.10.0                # Dataset loading and processing

# KNOWLEDGE GRAPH & SEMANTIC WEB
# ====================================================================
SPARQLWrapper>=2.0.0            # SPARQL query interface
networkx>=2.8                   # Graph analysis and manipulation

# VECTOR SEARCH & EMBEDDINGS
# ====================================================================
faiss-cpu>=1.7.4                # Facebook AI Similarity Search (CPU)
# faiss-gpu>=1.7.4              # Use this for GPU acceleration

# VISUALIZATION & PLOTTING
# ====================================================================
matplotlib>=3.5.0               # Plotting library
seaborn>=0.11.0                 # Statistical data visualization

# WEB & API INTEGRATION
# ====================================================================
requests>=2.28.0                # HTTP library
openai>=0.27.0                  # OpenAI API client

# UTILITIES & SYSTEM
# ====================================================================
tqdm>=4.64.0                    # Progress bars
logging>=0.4.9.6                # Logging utilities
datetime>=4.7                   # Date and time handling
typing-extensions>=4.3.0        # Type hints extensions
warnings>=0.1.0                 # Warning control
traceback2>=1.4.0               # Enhanced traceback

# DEVELOPMENT & TESTING (OPTIONAL)
# ====================================================================
# pytest>=7.0.0                 # Testing framework
# pytest-cov>=3.0.0             # Coverage testing
# black>=22.0.0                 # Code formatter
# flake8>=5.0.0                 # Code linting
# jupyter>=1.0.0                # Jupyter notebook support

# ====================================================================
# INSTALLATION INSTRUCTIONS
# ====================================================================
# 
# 1. Basic Installation:
#    pip install -r requirements.txt
# 
# 2. GPU Support (NVIDIA CUDA):
#    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#    pip install faiss-gpu
# 
# 3. Development Installation:
#    pip install -r requirements.txt
#    pip install pytest pytest-cov black flake8 jupyter
# 
# 4. Verify Installation:
#    python check_requirements.py
# 
# ====================================================================
# SYSTEM REQUIREMENTS
# ====================================================================
# 
# Minimum Requirements:
# - Python 3.8+
# - RAM: 8GB (16GB recommended)
# - Storage: 10GB free space
# - Internet connection for model downloads
# 
# Recommended for GPU:
# - NVIDIA GPU with CUDA 11.8+
# - VRAM: 8GB+ for large models
# ====================================================================
# CHIMERA-3X SYSTEM COMPONENTS
# ====================================================================
# 
# Core Modules:
# - X_main.py: Main medical expert system or the default
# - Chimera_3X_Benchmark.py: Comprehensive evaluation suite
# - Marwantoolkit/: Multi-modal AI toolkit (X1, X2, X3)
# - git_info/: Knowledge graph and retrieval systems
# - git_wiki_pub/: Publication and wiki integration
# 
# Key Features:
# - Multi-modal medical query processing
# - Advanced benchmarking (PubMedQA, BioASQ, Clinical Accuracy)
# - Knowledge graph integration
# - Safety detection and explainability
# - Comprehensive visualization dashboard
# 
# ====================================================================
# PERFORMANCE BENCHMARKS
# ====================================================================
# 
# Latest Test Results :
# - Overall Score: 77.47% (Grade: C)
# - PubMedQA: 84.04% (Target: 75%) ✓
# - BioASQ: 78.64% (Target: 70%) ✓
# - Clinical Accuracy: 80.45% (Target: 75%) ✓
# - Explainability: 59.15% (Target: 60%) ✗
# - Safety Detection: 78.12% (Target: 85%) ✗
# - Targets Achieved: 3/5 benchmarks
# 
# ====================================================================
# TROUBLESHOOTING
# ====================================================================
# 
# Common Issues:
# 
# 1. CUDA Out of Memory:
#    - Reduce batch size in generation_args
#    - Use CPU mode: device = "cpu"
#    - Enable gradient checkpointing
# 
# 2. Model Download Fails:
#    - Check internet connection
#    - Verify Hugging Face access token
#    - Try manual download: huggingface-cli download
# 
# 3. Import Errors:
#    - Run: pip install --upgrade -r requirements.txt
#    - Check Python version compatibility
#    - Verify virtual environment activation
# 
# 4. Performance Issues:
#    - Enable GPU acceleration
#    - Use quantization (bitsandbytes)
#    - Optimize generation parameters
# 
# ====================================================================
# CONTACT & SUPPORT
# ====================================================================
# 
# For issues and contributions:
# - GitHub Issues: https://github.com/Marwan/Chimera-3X/issues
# - Documentation: https://github.com/Marwan/Chimera-3X/wiki
# - Email: your-email@domain.com
