# Chimera-3X
A Hybrid Cognitive Engine for Retrieval-Augmented Biomedical Consulting
# Chimera-3X: Advanced Medical AI System
## Overview
Chimera-3X is a comprehensive medical AI system that combines multiple specialized models for enhanced medical text analysis, research, and consultation capabilities.

## Key Features
- Multi-Model Architecture (GPT-2, BioBERT, DeepSeek-R1, EXAONE-3.5)
- Medical Text Processing and Analysis
- Research Integration and Code Generation
- Safety Protocols and Validation
- Comprehensive Benchmarking System
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
- X1.py : Biomedical text toolkit (GPT-2 + BioBERT)
- X2.py : Research assistant (DeepSeek-R1)
- X3.py : Medical chat assistant (EXAONE-3.5)
- Chimera_3X_Benchmark.py : Performance evaluation system
## Performance Metrics
- EM Score : 0.8404
- F1 Score : 0.8045
- Clinical Accuracy : 4.02/5
- Explainability : 2.96/5
- Trustworthiness : 3.9/5
## Requirements
- Python 3.8+
- PyTorch
- Transformers
- OpenAI API key
- CUDA (recommended)
## Project Structure
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
⚠️ Medical Disclaimer : This system is for research and educational purposes only. Not intended for actual medical diagnosis or treatment decisions.

## License
MIT License

## Citation
@software{chimera3x2024,
  title={Chimera-3X: Advanced Medical AI System},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[Marwan]/Chimera-3X}
}

## Contact
- Developer: [Your Name]
- Email: [Your Email]
- GitHub: [Your GitHub Profile]
