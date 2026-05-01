
# Chimera-3X : A Hybrid Cognitive Engine for Retrieval-Augmented Biomedical Consulting


Chimera-3X is an advanced biomedical question answering system designed to tackle the complexity of clinical knowledge with high accuracy and reliability. It integrates multiple specialized AI components and external biomedical knowledge graphs to enhance factual correctness and reasoning.

<img width="2535" height="1320" alt="image" src="https://github.com/user-attachments/assets/17281228-e85d-4e8e-bbdc-f8c13bd596ea" />


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


<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/4708df72-fb9b-4ea2-ac77-17d8ac31aca7" width="370"/>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/c3bac357-0757-49fd-81e4-1eeb07bf4bd6" width="285"/>
    </td>
  </tr>
</table>




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
  url = { https://ieeexplore.ieee.org/document/11356117 }
  
}


## Contact
- Developer: [Mr.Marwan]
- Email: [marwan@mail.dlut.edu.cn]
- ORCID ID: [0009-0003-9052-6873]
- pdf : https://ieeexplore.ieee.org/document/11356117
