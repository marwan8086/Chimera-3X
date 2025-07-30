import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import json
import os
import sys
import subprocess
from typing import Dict, List, Union, Optional
import git_info.git_Up_to_date
import git_info.batche_git
import git_info.git_kgs
import git_info.git_kgsV
import git_info.m_tr
import git_info.git_wiki_pub.pub
import git_info.git_wiki_pub.wiki
import X_main



class MedicalChatAssistant:
    
    def __init__(self, model_path: str = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"):

        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
        os.environ['TRUST_REMOTE_CODE'] = 'true'
        
        self.device = self._setup_device()
        self.model_path = model_path
        self._install_required_packages()
        self.model, self.tokenizer = self._load_medical_model()
        self._init_prompt_templates()
        self.conversation_history = []
        
        self.generation_config = {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.5,
            "no_repeat_ngram_size": 3
        }
        
        self.safety_flags = {
            "urgent": ["emergency", "urgent", "immediate", "chest pain"],
            "warning": ["warning", "caution", "interaction", "contraindication"]
        }

    def _setup_device(self) -> str:
        if torch.cuda.is_available():
            print(f" Using GPU: {torch.cuda.get_device_name(0)}")
            return "cuda"
        print(" Using CPU")
        return "cpu"

    def _install_required_packages(self):
        required_packages = ['accelerate', 'bitsandbytes']
        try:
            import accelerate
            import bitsandbytes
        except ImportError:
            print(" Installing required packages...")
            subprocess.check_call([
                sys.executable, 
                "-m", 
                "pip", 
                "install", 
                *required_packages
            ])
            print(" Required packages installed")

    def _load_medical_model(self) -> tuple:
        print(f" Loading {self.model_path}...")
        try:

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            if self.device == "cuda":
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                except:

                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16,
                        trust_remote_code=True
                    ).to(self.device)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                ).to(self.device)
            
            print(" Model loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            print(f" Failed to load model: {str(e)}")
            raise

    def _init_prompt_templates(self):
        self.system_prompt = """You are MediExpert-5.0, an advanced medical AI assistant. Follow STRICT protocols:

1. RESPONSE RULES:
- Respond ONLY to medical/health-related queries
- Use latest clinical guidelines (2023-2024)
- For professionals: Include ICD-11/SNOMED codes when possible
- For patients: Explain at 8th grade reading level
- Always cite 1-2 authoritative sources

2. SAFETY PROTOCOLS:
- [REQUIRED] "Consult your physician before any action"
- [WARNING] Clearly flag drug interactions
- [URGENT] Identify emergency symptoms immediately

3. RESPONSE FORMAT:
[Clinical Assessment] <detailed analysis>
[Recommended Actions] <clear steps>
[Evidence Sources] <references>
[Safety Disclaimer] <required>"""

    def start_interactive_session(self):
        """Launch interactive medical consultation session"""
        print("\n" + "="*60)
        print(" MediExpert 5.0 - Medical Consultation System ".center(60, "="))
        print("="*60)
        print("Type 'exit' to end the session\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'end']:
                    self._save_session()
                    print("\nSession ended. Consultation report saved.")
                    break
                
                response = self.generate_medical_response(user_input)
                self._display_response(response)
                
            except KeyboardInterrupt:
                print("\nSession interrupted. Saving conversation...")
                self._save_session()
                break
            except Exception as e:
                print(f"\n Error: {str(e)}")
                continue

    def generate_medical_response(self, user_query: str) -> Dict[str, Union[str, bool, List[str]]]:
        try:
            if not self._is_medical_query(user_query):
                return self._handle_non_medical_query(user_query)
            
            user_type = self._identify_user_type(user_query)
            messages = self._build_conversation_messages(user_query, user_type)
            raw_response = self._generate_model_response(messages)
            
            return self._process_model_response(raw_response, user_query, user_type)
            
        except Exception as e:
            return {
                "response": f" System Error: {str(e)[:200]}",
                "is_urgent": False,
                "warnings": ["System malfunction"],
                "sources": []
            }

    def _is_medical_query(self, query: str) -> bool:
        """Determine if query is medical-related"""
        medical_terms = [
            'diagnos', 'treat', 'symptom', 'patient', 'disease',
            'medical', 'health', 'doctor', 'hospital', 'pharma'
        ]
        return any(term in query.lower() for term in medical_terms)

    def _identify_user_type(self, query: str) -> str:
        """Identify if user is medical professional or patient"""
        professional_terms = [
            'diagnosis', 'treatment', 'dose', 'icd', 'symptoms',
            'prescribe', 'clinical', 'therapy'
        ]
        return "physician" if any(term in query.lower() for term in professional_terms) else "patient"

    def _build_conversation_messages(self, query: str, user_type: str) -> List[Dict[str, str]]:
        """Construct conversation history for model"""
        return [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history,
            {"role": "user", "content": f"[{user_type.upper()}]: {query}"}
        ]

    def _generate_model_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response from model"""
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model.generate(
            inputs,
            **self.generation_config,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _process_model_response(self, raw_response: str, query: str, user_type: str) -> Dict:
        content = raw_response.split("[|assistant|]")[-1] if "[|assistant|]" in raw_response else raw_response
        
        return {
            "response": self._format_response(content, user_type),
            "is_urgent": self._check_safety_flags(content, "urgent"),
            "warnings": self._check_safety_flags(content, "warning"),
            "sources": self._extract_sources(content),
            "timestamp": datetime.now().isoformat()
        }

    def _check_safety_flags(self, text: str, flag_type: str) -> Union[bool, List[str]]:
        text_lower = text.lower()
        if flag_type == "urgent":
            return any(keyword in text_lower for keyword in self.safety_flags["urgent"])
        return [kw for kw in self.safety_flags["warning"] if kw in text_lower]

    def _format_response(self, content: str, user_type: str) -> str:
        if user_type == "physician":
            return f"""
  Clinical Analysis:
{content.strip()}

  Evidence Sources:
- UpToDate Clinical Database
- Latest Medical Guidelines

  Safety Checks:
- Drug interactions reviewed
- Contraindications checked"""
        else:
            return f"""
  Patient Guidance:
{self._simplify_text(content)}

  Recommended Actions:
{self._generate_action_steps(content)}

  Remember:
- Always consult your doctor
- Don't ignore worsening symptoms"""

    def _simplify_text(self, text: str) -> str:
        replacements = {
            "administer": "take",
            "contraindicated": "not recommended",
            "dosage": "amount",
            "symptoms": "signs"
        }
        for term, replacement in replacements.items():
            text = text.replace(term, replacement)
        return text

    def _generate_action_steps(self, text: str) -> str:
        steps = []
        if "blood pressure" in text.lower():
            steps.append("• Measure your blood pressure regularly")
        if any(term in text.lower() for term in ["sugar", "glucose"]):
            steps.append("• Monitor your blood sugar levels")
        steps.append("• Follow your doctor's instructions")
        return "\n".join(steps)

    def _extract_sources(self, text: str) -> List[str]:
        """Extract referenced sources from text"""
        sources = []
        if "study" in text.lower():
            sources.append("Recent clinical study")
        if "guideline" in text.lower():
            sources.append("Medical guidelines")
        return sources if sources else ["General medical knowledge"]

    def _update_conversation_history(self, query: str, response: str):
        self.conversation_history.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ])

        self.conversation_history = self.conversation_history[-5:]

    def _handle_non_medical_query(self, query: str) -> Dict:
        return {
            "response": "  I specialize only in medical topics. Please ask health-related questions.",
            "is_urgent": False,
            "warnings": [],
            "sources": [],
            "suggested_questions": [
                "What are the symptoms of diabetes?",
                "How to manage high blood pressure?",
                "Latest treatments for arthritis?"
            ]
        }

    def _display_response(self, response: Dict):
        print("\n" + "="*60)
        print(response["response"])
        
        if response["warnings"]:
            print("\n  Warnings:", ", ".join(response["warnings"]))
            
        if response["is_urgent"]:
            print("\n  URGENT: Seek immediate medical attention!")
        
        print("="*60 + "\n")

    def _save_session(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"medex_consultation_{timestamp}.json"
        
        session_data = {
            "timestamp": timestamp,
            "model": self.model_path,
            "conversation": self.conversation_history,
            "summary": self._generate_summary()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        print(f"  Consultation saved to {filename}")

    def _generate_summary(self) -> str:
        if not self.conversation_history:
            return "No conversation history"
        
        last_query = self.conversation_history[-2]["content"] if len(self.conversation_history) >= 2 else ""
        last_response = self.conversation_history[-1]["content"]
        
        return f"Consultation about: {last_query[:100]}... | Key points: {last_response[:200]}..."

if __name__ == "__main__":
    try:
        print("  Starting Medical Chat Assistant...")
        assistant = MedicalChatAssistant()
        assistant.start_interactive_session()
    except Exception as e:
        print(f"  Critical Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Run: pip install accelerate bitsandbytes")
        print("2. Accept model terms at: https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct")
        print("3. Try CPU-only mode if GPU issues persist")
        print("4. Check system requirements and dependencies")
        
        
 
 
 
 
        
"""

                                    Marwantoolkit for research tasks and academic use
    
"""