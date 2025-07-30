from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel, GPT2Tokenizer
import torch
import warnings
import re
import os
import git_info.git_Up_to_date
import git_info.batche_git
import git_info.git_kgs
import git_info.git_kgsV
import git_info.m_tr
import git_info.git_wiki_pub.pub
import git_info.git_wiki_pub.wiki
import X_main



warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

class BiomedicalTextToolkit:
    def __init__(self):
        print("Loading biomedical models...")
        
        self.medical_generator = self._load_medical_generator()
        
        self.text_analyzer = self._load_text_analyzer()
        
        print(" All models loaded successfully!")
    
    def _load_medical_generator(self):
        try:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            return {"tokenizer": tokenizer, "model": model}
        except Exception as e:
            print(f" Error loading medical generator: {e}")
            return None
    
    def _load_text_analyzer(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
            model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
            return {"tokenizer": tokenizer, "model": model}
        except Exception as e:
            print(f" Error loading text analyzer: {e}")
            return None
    
    def _clean_generated_text(self, text):

        text = re.sub(r'\([^)]*\)', '', text)  
        text = re.sub(r'\[[^\]]*\]', '', text)  
        text = re.sub(r'Dr\. [A-Z][a-z]+ [A-Z][a-z]+', 'medical professionals', text)  
        text = re.sub(r'University[^.]*\.', '', text)  
        text = re.sub(r'Figure \d+', '', text)  
        text = re.sub(r'Table \d+', '', text)  
        text = re.sub(r'\s+', ' ', text)  
        text = re.sub(r'[.,]\s*[.,]', '.', text)  
        
        sentences = text.split('.')
        clean_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if (sentence and 
                len(sentence) > 15 and 
                not any(artifact in sentence.lower() for artifact in 
                       ['wikipedia', 'footnote', 'see section', 'reference', 'citation', 
                        'press release', 'said dr', 'according to', 'reported by'])):
                clean_sentences.append(sentence)
        
        if clean_sentences:
            result = '. '.join(clean_sentences[:2])  
            if not result.endswith('.'):
                result += '.'
            return result
        
        return "Medical information requires professional verification."
    
    def generate_medical_text(self, prompt, max_length=100, temperature=0.7, style="clinical"):
        if self.medical_generator is None:
            return " Medical generator not available"
        
        medical_contexts = {
            "clinical": "In clinical practice, ",
            "general": "Medically, ",
            "definition": "Medical definition: ",
            "symptoms": "Common symptoms include: ",
            "treatment": "Treatment typically involves: ",
            "causes": "Medical causes include: "
        }
        
        prompt_lower = prompt.lower()
        if "causes" in prompt_lower or "caused by" in prompt_lower:
            context = medical_contexts["causes"]
        elif "symptoms" in prompt_lower or "signs" in prompt_lower:
            context = medical_contexts["symptoms"]
        elif "treatment" in prompt_lower or "therapy" in prompt_lower:
            context = medical_contexts["treatment"]
        elif "definition" in prompt_lower or "is defined as" in prompt_lower:
            context = medical_contexts["definition"]
        else:
            context = medical_contexts.get(style, medical_contexts["clinical"])
        
        full_prompt = context + prompt
        
        tokenizer = self.medical_generator["tokenizer"]
        model = self.medical_generator["model"]
        
        try:
            inputs = tokenizer(
                full_prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=200
            )
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=min(max_length + len(inputs.input_ids[0]), 200),
                    temperature=max(0.4, min(temperature, 0.8)),
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=1
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = generated_text.replace(context, "").strip()
           
            return self._clean_generated_text(result)
            
        except Exception as e:
            return f" Error generating text: {e}"
    
    def analyze_medical_similarity(self, text1, text2):
      
        if self.text_analyzer is None:
            return 0.0
        
        tokenizer = self.text_analyzer["tokenizer"]
        model = self.text_analyzer["model"]
        
        try:
            inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs1 = model(**inputs1)
                outputs2 = model(**inputs2)
            
            emb1 = outputs1.last_hidden_state.mean(dim=1)
            emb2 = outputs2.last_hidden_state.mean(dim=1)
            
            similarity = torch.cosine_similarity(emb1, emb2)
            return float(similarity.item())
            
        except Exception as e:
            print(f" Error calculating similarity: {e}")
            return 0.0
    
    def find_similar_symptoms(self, target_symptom, symptom_list):
      
        similarities = []
        
        for symptom in symptom_list:
            try:
                similarity = self.analyze_medical_similarity(target_symptom, symptom)
                similarities.append((symptom, similarity))
            except Exception as e:
                print(f" Error comparing '{symptom}': {e}")
                similarities.append((symptom, 0.0))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def generate_patient_summary(self, symptoms, treatments, max_length=120):
        symptoms_str = ", ".join(symptoms)
        treatments_str = ", ".join(treatments)
        
        prompt = f"Patient presents with {symptoms_str}. Treatment includes {treatments_str}. Assessment shows"
        
        return self.generate_medical_text(prompt, max_length=max_length, temperature=0.5, style="clinical")
    
    def explain_medical_term(self, term, max_length=80):
        
        medical_definitions = {
            "diabetes": "a metabolic disorder characterized by high blood sugar levels due to insulin deficiency or resistance",
            "hypertension": "a condition where blood pressure remains consistently elevated above normal levels",
            "pneumonia": "an infection that inflames air sacs in one or both lungs, which may fill with fluid",
            "asthma": "a chronic respiratory condition causing inflammation and narrowing of the airways",
            "stroke": "a medical emergency where blood supply to part of the brain is interrupted",
            "heart attack": "occurs when blood flow to the heart muscle is blocked, usually by a blood clot"
        }
        
        if term.lower() in medical_definitions:
            return f"{term.title()} is {medical_definitions[term.lower()]}."
        else:
            prompt = f"{term.title()} is a medical condition characterized by"
            return self.generate_medical_text(prompt, max_length=max_length, temperature=0.4, style="definition")
    
    def validate_medical_text(self, text):
        issues = []
        score = 100
       
        artifacts = ['dr.', 'university', 'press release', 'figure', 'table', 'said', 'according to']
        for artifact in artifacts:
            if artifact in text.lower():
                issues.append(f"Contains artifact: {artifact}")
                score -= 15
      
        if len(text.split()) < 5:
            issues.append("Text too short")
            score -= 20
        
        medical_terms = ['medical', 'patient', 'treatment', 'symptom', 'condition', 'disease', 'clinical', 'therapy']
        if not any(term in text.lower() for term in medical_terms):
            issues.append("Lacks medical terminology")
            score -= 15
        
        if text.count('.') > 3:
            issues.append("Too many sentences")
            score -= 10
        
        return {
            'score': max(0, score),
            'issues': issues,
            'is_valid': score > 60
        }
    
    def generate_validated_medical_text(self, prompt, max_length=100, temperature=0.7, style="clinical", max_attempts=3):
        best_result = None
        best_score = 0
        
        for attempt in range(max_attempts):
            current_temp = max(0.4, temperature - (attempt * 0.1))
            
            generated_text = self.generate_medical_text(prompt, max_length, current_temp, style)
            validation = self.validate_medical_text(generated_text)
            
            if validation['score'] > best_score:
                best_score = validation['score']
                best_result = {
                    'text': generated_text,
                    'validation': validation,
                    'attempt': attempt + 1
                }
            
            if validation['score'] > 80:
                break
        
        return best_result or {
            'text': "Unable to generate quality medical text",
            'validation': {'score': 0, 'issues': ['Generation failed'], 'is_valid': False},
            'attempt': max_attempts
        }


_toolkit_instance = None

def get_toolkit():
    global _toolkit_instance
    if _toolkit_instance is None:
        _toolkit_instance = BiomedicalTextToolkit()
    return _toolkit_instance

def demonstrate_toolkit():
    print("=== Enhanced Biomedical Text Toolkit Demo ===\n")
    
    toolkit = get_toolkit()
    
    print("1. Medical Text Generation:")
    print("-" * 50)
    
    conditions = ["diabetes mellitus", "hypertension", "pneumonia"]
    
    for condition in conditions:
        explanation = toolkit.explain_medical_term(condition, max_length=100)
        print(f" {condition.title()}:")
        print(f"   {explanation}\n")
    
    print("2. Symptom Similarity Analysis:")
    print("-" * 50)
    
    target = "chest pain"
    symptoms = [
        "shortness of breath",
        "headache", 
        "abdominal pain",
        "difficulty breathing",
        "nausea",
        "heart palpitations",
        "dizziness"
    ]
    
    print(f" Symptoms similar to '{target}':")
    similar_symptoms = toolkit.find_similar_symptoms(target, symptoms)
    for symptom, score in similar_symptoms:
        print(f"   • {symptom}: {score:.3f}")
    
    print("\n3. Patient Summary Generation:")
    print("-" * 50)
    
    patient_symptoms = ["fever", "persistent cough", "fatigue", "shortness of breath"]
    patient_treatments = ["rest", "increased fluid intake", "acetaminophen", "monitoring"]
    
    summary = toolkit.generate_patient_summary(patient_symptoms, patient_treatments, max_length=150)
    print(f" Patient Summary:")
    print(f"   {summary}")
    
    print("\n4. Validated Medical Text Generation:")
    print("-" * 50)
    
    prompts = [
        "The main causes of diabetes include",
        "Symptoms of pneumonia typically involve", 
        "Treatment for hypertension usually consists of"
    ]
    
    for prompt in prompts:
        result = toolkit.generate_validated_medical_text(prompt, max_length=80, temperature=0.6)
        print(f"\n Prompt: {prompt}")
        print(f"   Generated: {result['text']}")
        print(f"   Quality Score: {result['validation']['score']}/100")
        if result['validation']['issues']:
            print(f"   Issues: {', '.join(result['validation']['issues'])}")
    
    print("\n5. Medical Text Quality Analysis:")
    print("-" * 50)
    
    sample_text = "The patient presented with acute chest pain and shortness of breath. Initial assessment revealed elevated heart rate and blood pressure. Treatment included oxygen therapy and cardiac monitoring."
    
    validation = toolkit.validate_medical_text(sample_text)
    print(f" Sample Text Quality:")
    print(f"   Score: {validation['score']}/100")
    print(f"   Valid: {validation['is_valid']}")
    if validation['issues']:
        print(f"   Issues: {', '.join(validation['issues'])}")


def quick_medical_generation(prompt, length=100):
    toolkit = get_toolkit()
    return toolkit.generate_medical_text(prompt, max_length=length)

def quick_similarity_check(text1, text2):
    toolkit = get_toolkit()
    return toolkit.analyze_medical_similarity(text1, text2)

def batch_similarity_analysis(target_text, text_list):
    toolkit = get_toolkit()
    results = []
    
    for i, text in enumerate(text_list):
        similarity = toolkit.analyze_medical_similarity(target_text, text)
        results.append({
            'index': i,
            'text': text,
            'similarity': similarity
        })
    
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results

if __name__ == "__main__":
    try:
        
        demonstrate_toolkit()
        
        print("\n" + "="*70)
        print(" Quick Usage.....")
        print("="*70)
        
    
        toolkit = get_toolkit()
        
        print("Created by Marwan just for Research........")
        print("_____________________________ Marwantoolkit___________________________")
        
        print("\n Generating medical text...")
        result1 = quick_medical_generation("Common symptoms of COVID-19 include", 80)
        print(f"Generated: {result1}")
        
        print("\n Medical term explanation...")
        diabetes_explanation = toolkit.explain_medical_term("diabetes", 100)
        print(f"Diabetes explanation: {diabetes_explanation}")
        
        print("\n Checking similarity...")
        similarity = quick_similarity_check("high blood pressure", "hypertension")
        print(f"Similarity between 'high blood pressure' and 'hypertension': {similarity:.3f}")
        
        print("\n Batch similarity analysis...")
        target = "heart disease"
        conditions = ["cardiac arrest", "diabetes", "hypertension", "pneumonia", "stroke"]
        
        batch_results = batch_similarity_analysis(target, conditions)
        print(f"Conditions most similar to '{target}':")
        for result in batch_results[:3]:
            print(f"   • {result['text']}: {result['similarity']:.3f}")
        
        print("\n Patient summary example...")
        sample_symptoms = ["headache", "fever", "nausea"]
        sample_treatments = ["rest", "hydration", "pain medication"]
        summary = toolkit.generate_patient_summary(sample_symptoms, sample_treatments, 100)
        print(f"Patient summary: {summary}")
        
        print("\n Demo completed successfully!")
        
    except Exception as e:
        print(f" Error during demonstration: {e}")
        print("Please check your transformers installation and internet connection.")
        
        
        
        
        
"""

                                    Marwantoolkit for research tasks and academic use
    
"""