import git_info.git_Up_to_date
import git_info.batche_git
import git_info.git_kgs
import git_info.git_kgsV
import git_info.m_tr
import git_info.git_wiki_pub.pub
import git_info.git_wiki_pub.wiki
import X_main
import requests
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deepseek_research.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeepSeekResearcher:
     
    def __init__(self):
 
        self.api_key = "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" # Just replace your API here
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "deepseek/deepseek-r1" 

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/marwan-research",
            "X-Title": "DeepSeek Research Client"
        }
        
        self.default_params = {
            "model": self.model,
            "temperature": 0.1,
            "max_tokens": 4000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_history = []
        
        logger.info(f" DeepSeek-R1 Research Client initialized - Session: {self.session_id}")
    
    def _make_request(self, messages: List[Dict], **kwargs) -> Dict:
     
        params = {**self.default_params, **kwargs}
        params["messages"] = messages
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=params,
                timeout=120
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise Exception(" Authentication failed. Please check your API key.")
            elif response.status_code == 429:
                raise Exception(" Rate limit exceeded. Please try again later.")
            elif response.status_code == 400:
                raise Exception(" Bad request. Please check your input parameters.")
            else:
                raise Exception(f" HTTP Error {response.status_code}: {e}")
        except requests.exceptions.RequestException as e:
            raise Exception(f" Request failed: {e}")
    
    def ask(self, question: str, system_prompt: Optional[str] = None, **kwargs) -> str:

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": question})
        
        try:
            response = self._make_request(messages, **kwargs)
            answer = response['choices'][0]['message']['content']
            
            # Log the interaction
            interaction = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer": answer,
                "tokens_used": response.get('usage', {}).get('total_tokens', 0)
            }
            self.conversation_history.append(interaction)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in ask(): {e}")
            return f" Error: {str(e)}"
    
    def reasoning_analysis(self, problem: str, domain: str = "general") -> str:
   
        system_prompt = f"""You are DeepSeek-R1, an advanced reasoning model.
        
        Task: Analyze the following {domain} problem using systematic reasoning.
        
        Instructions:
        1. Break down the problem into clear components
        2. Show your step-by-step reasoning process
        3. Identify key concepts and relationships
        4. Provide detailed explanations for each step
        5. Double-check your reasoning for accuracy
        6. Present the final solution clearly
        
        Think carefully and show your complete reasoning process."""
        
        return self.ask(problem, system_prompt, temperature=0.05, max_tokens=4000)
    
    def code_generation(self, task: str, language: str = "python", style: str = "clean") -> str:
   
        system_prompt = f"""You are DeepSeek-R1, a coding assistant with advanced reasoning.
        
        Task: Generate {language} code for the following task.
        
        Requirements:
        1. Write {style} and well-documented code
        2. Include comprehensive comments
        3. Explain your approach and reasoning
        4. Provide usage examples
        5. Consider edge cases and error handling
        6. Include unit tests if appropriate
        
        Focus on producing high-quality, maintainable code with clear explanations."""
        
        return self.ask(task, system_prompt, temperature=0.2, max_tokens=4000)
    
    def research_query(self, query: str, field: str = "general", depth: str = "comprehensive") -> str:
    
        system_prompt = f"""You are DeepSeek-R1, a research assistant specialized in {field}.
        
        Task: Provide a {depth} analysis of the following research question.
        
        Instructions:
        1. Analyze the question from multiple angles
        2. Provide current understanding and key concepts
        3. Discuss relevant theories and methodologies
        4. Identify knowledge gaps and open questions
        5. Suggest future research directions
        6. Structure your response clearly with headings
        
        Provide an academic-level analysis with proper reasoning."""
        
        return self.ask(query, system_prompt, temperature=0.3, max_tokens=4000)
    
    def comparative_analysis(self, topic1: str, topic2: str, context: str = "general") -> str:
   
        query = f"""Compare and contrast {topic1} and {topic2} in the context of {context}.
        
        Please provide:
        1. Key similarities between the topics
        2. Major differences
        3. Advantages and disadvantages of each
        4. Use cases and applications
        5. Future prospects
        6. Recommendations based on your analysis"""
        
        return self.research_query(query, context, "comprehensive")
    
    def batch_analysis(self, questions: List[str], delay: float = 2.0) -> List[Dict]:
    
        results = []
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
            
            try:
                answer = self.ask(question)
                results.append({
                    "question": question,
                    "answer": answer,
                    "index": i,
                    "status": "success"
                })
                
                # Add delay to avoid rate limiting
                if i < len(questions) - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {e}")
                results.append({
                    "question": question,
                    "error": str(e),
                    "index": i,
                    "status": "error"
                })
        
        return results
    
    def save_session(self, filename: Optional[str] = None):
        """
        Save conversation history to file
        
        Args:
            filename: Optional filename (auto-generated if not provided)
        """
        if not filename:
            filename = f"research_session_{self.session_id}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "session_id": self.session_id,
                    "timestamp": datetime.now().isoformat(),
                    "total_interactions": len(self.conversation_history),
                    "conversation_history": self.conversation_history
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f" Session saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            return None
    
    def get_session_stats(self) -> Dict:

        total_tokens = sum(interaction.get('tokens_used', 0) for interaction in self.conversation_history)
        
        return {
            "session_id": self.session_id,
            "total_interactions": len(self.conversation_history),
            "total_tokens_used": total_tokens,
            "average_tokens_per_interaction": total_tokens / len(self.conversation_history) if self.conversation_history else 0,
            "session_duration": datetime.now().isoformat()
        }

def test_connection():
    """Test API connection"""
    print(" Testing DeepSeek-R1 connection...")
    
    try:
        client = DeepSeekResearcher()
        response = client.ask("Hello! Please respond with 'Connection successful' to confirm API is working.")
        
        if "Connection successful" in response or "successful" in response.lower():
            print(" API connection successful!")
            print(f"Response: {response}")
            return True
        else:
            print(f"  Unexpected response: {response}")
            return False
            
    except Exception as e:
        print(f" Connection failed: {e}")
        return False

def run_research_examples():
    """Run comprehensive research examples"""
    
    client = DeepSeekResearcher()
    
    print("\n" + "="*60)
    print(" MATHEMATICAL REASONING EXAMPLE")
    print("="*60)
    
    math_problem = """
    A researcher has a dataset with 1000 samples. She uses 60% for training, 
    20% for validation, and 20% for testing. During cross-validation, 
    she uses 5-fold validation on the training set. 
    How many samples are used in each fold for training and validation?
    """
    
    result = client.reasoning_analysis(math_problem, "mathematics")
    print(result)
    
    print("\n" + "="*60)
    print(" CODE GENERATION EXAMPLE")
    print("="*60)
    
    code_task = """
    Create a Python class for a research data processor that can:
    1. Load CSV files
    2. Handle missing values
    3. Perform basic statistical analysis
    4. Export results to different formats
    """
    
    result = client.code_generation(code_task, "python", "research")
    print(result)
    
    print("\n" + "="*60)
    print(" RESEARCH ANALYSIS EXAMPLE")
    print("="*60)
    
    research_question = """
    What are the current challenges and opportunities in 
    applying large language models for scientific research?
    """
    
    result = client.research_query(research_question, "artificial intelligence", "comprehensive")
    print(result)
    
    print("\n" + "="*60)
    print(" COMPARATIVE ANALYSIS EXAMPLE")
    print("="*60)
    
    comparison = client.comparative_analysis(
        "Supervised Learning", 
        "Unsupervised Learning", 
        "machine learning research"
    )
    print(comparison)

    filename = client.save_session()
    
    stats = client.get_session_stats()
    print("\n" + "="*60)
    print(" SESSION STATISTICS")
    print("="*60)
    print(f"Session ID: {stats['session_id']}")
    print(f"Total Interactions: {stats['total_interactions']}")
    print(f"Total Tokens Used: {stats['total_tokens_used']}")
    print(f"Average Tokens per Interaction: {stats['average_tokens_per_interaction']:.2f}")
    print(f"Session saved to: {filename}")

def main():
    print(" _________________________________Marwantoolkit___________________________")
    print("="*60)
    print("Created by Marwan just for Research ........")
    print("="*60)
    
    if test_connection():
        print("\n Running research examples...")
        run_research_examples()
    else:
        print("\n Connection failed. Please check your setup.")

if __name__ == "__main__":
    main()
    



        
"""

                                    Marwantoolkit for research tasks and academic use
    
"""