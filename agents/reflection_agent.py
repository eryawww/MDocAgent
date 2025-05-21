from agents.multi_agent_system import MultiAgentSystem
from tqdm import tqdm
import torch
import json
from mydatasets.base_dataset import BaseDataset
import hydra
import os
from typing import List, Tuple, Dict, Any, Optional, Union
from models.base_model import BaseModel
import importlib

class ReflectionAgent(MultiAgentSystem):
    """
    A reflection-based agent that generates and iteratively improves answers through structured planning and technical analysis.
    
    The agent works in the following way:
    1. Planning Phase (First Iteration):
       - Creates a detailed plan before generating the initial answer
       - Identifies key technical components and concepts
       - Determines required information from context
       - Structures the answer approach
       - Anticipates potential challenges
       - Assesses confidence in proceeding with the answer
    
    2. Initial Answer Generation:
       - Generates a comprehensive answer based on the plan
       - Ensures technical accuracy and precision
       - Maintains clear explanation of complex concepts
       - Uses proper technical terminology
       - Follows logical structure
    
    3. Reflection and Improvement:
       - Performs detailed technical analysis of the answer
       - Evaluates technical accuracy and correctness
       - Checks completeness of technical details
       - Assesses precision of explanations
       - Identifies areas for improvement
       - Generates improved versions with confidence scoring
    
    4. Iteration Control:
       - Continues reflection until either:
         * Maximum iterations reached (config.max_reflection_iter)
         * Confidence threshold achieved (config.confidence_threshold)
    
    The agent processes both text and image inputs, maintaining a complete history of all
    planning, generation, and reflection steps for analysis and debugging purposes.
    """
    def __init__(self, config: Any):
        """
        Initialize the ReflectionAgent with the given configuration.
        
        Args:
            config: Configuration object containing parameters such as:
                   - max_reflection_iter: Maximum number of reflection iterations
                   - confidence_threshold: Confidence threshold to stop reflection
                   - ans_key: Key for storing answers in the dataset
                   - save_message: Whether to save messages in the dataset
                   - save_freq: Frequency of saving results
        """
        self.config = config
        self.agents = []
        self.models:dict = {}
        for agent_config in self.config.agents:
            if agent_config.model.class_name not in self.models:
                module = importlib.import_module(agent_config.model.module_name)
                model_class = getattr(module, agent_config.model.class_name)
                print("Create model: ", agent_config.model.class_name)
                self.models[agent_config.model.class_name] = model_class(agent_config.model)
            self.add_agent(agent_config, self.models[agent_config.model.class_name])
    
    def extract_confidence(self, message: str) -> float:
        """
        Extract confidence score from agent's message.
        
        Args:
            message: The message to extract confidence from
            
        Returns:
            float: Extracted confidence score (0.0 to 1.0), or 0.0 if not found
        """
        try:
            # Case insensitive search for confidence
            lower_message = message.lower()
            if "confidence:" in lower_message:
                # Extract the part after "confidence:"
                confidence_part = lower_message.split("confidence:")[1].strip()
                # Take the first word and convert to float
                confidence_str = confidence_part.split()[0].strip()
                confidence = float(confidence_str)
                
                # Ensure confidence is in valid range
                return max(0.0, min(1.0, confidence))
            else:
                return 0.0
        except Exception as e:
            print(f"Error extracting confidence: {e}")
            return 0.0
    
    def get_initial_prompt(self, question: str, plan: str = None) -> str:
        """
        Generate the initial prompt for either planning or first answer generation.
        """
        if plan is None:
            # Planning prompt
            return f"""Create a structured plan for answering: {question}

            1. Key Components: Main technical concepts, relationships
            2. Information Requirements: Relevant context from texts/images
            3. Answer Structure: Organization, technical detail level
            4. Potential Challenges: Complex aspects, assumptions, possible confusion

            End with "Confidence: X.X" (0.0-1.0)
            """
        # First answer prompt
        return f"""Answer based on this plan:

            Question: {question}

            Plan: {plan}

            Ensure: Technical accuracy, clear explanations, proper terminology, logical flow.
            End with "Confidence: X.X" (0.0-1.0)
            """

    def get_reflection_prompt(self, question: str, plan: str, current_ans: str) -> str:
        """
        Generate the reflection prompt for analyzing and improving the answer.
        """
        return f"""Analyze and improve my previous answer:

            Question: {question}
            Plan: {plan}
            Previous answer: {current_ans}

            Technical analysis:
            1. Accuracy: Check for technical errors, term usage, appropriate depth
            2. Completeness: Identify missing details, components, dependencies
            3. Precision: Areas for more precise explanation, terminology, relationships
            4. Information: Additional technical information needed

            Provide improved answer if needed or confirm if original is sufficient.
            End with "Confidence: X.X" (0.0-1.0)
            """

    def predict(self, question: str, texts: List[str], images: List[str]) -> Tuple[str, List[Dict]]:
        """
        Generate a response to the given question using a reflective approach.
        
        Args:
            question: The question to answer
            texts: List of text passages for context
            images: List of image paths for context
            
        Returns:
            Tuple containing:
            - The final answer string
            - List of message dictionaries containing the entire conversation history
        """
        # Initialize with general agent
        general_agent: BaseModel = self.agents[0]

        current_iter, current_confidence = 0, 0.0
        all_messages = []
        current_ans = ""  # Initialize current_ans
        plan = None

        while current_iter < self.config.max_reflection_iter:
            print(f"Iteration {current_iter}")
            if current_iter == 0:
                # First iteration: Planning phase
                planning_prompt = self.get_initial_prompt(question)
                plan, plan_messages = general_agent.predict(planning_prompt, texts, images)
                print('='*20, 'plan: \n', plan)
                all_messages.extend(plan_messages)
                
                # Generate initial answer based on the plan
                answer_prompt = self.get_initial_prompt(question, plan)
                print('='*20, 'answer_prompt: \n', answer_prompt)
                current_ans, messages = general_agent.predict(answer_prompt, texts, images)
            else:
                # Reflection phase
                reflection_prompt = self.get_reflection_prompt(question, plan, current_ans)
                print('='*20, 'reflection_prompt: \n', reflection_prompt)
                current_ans, messages = general_agent.predict(reflection_prompt, texts, images)

            print('='*20, 'current_ans: \n', current_ans)
            print('='*20, 'messages: \n', messages)
            all_messages.extend(messages)
            current_confidence = self.extract_confidence(current_ans)

            # Remove confidence score from the answer
            if "confidence:" in current_ans.lower():
                current_ans = current_ans.lower().split("confidence:")[0].strip()
            
            # Check if we should stop reflecting
            if current_confidence >= self.config.confidence_threshold:
                break
            
            current_iter += 1
                
        return current_ans, all_messages

    def predict_dataset(self, dataset:BaseDataset, resume_path = None):
        samples = dataset.load_data(use_retreival=True)
        if resume_path:
            assert os.path.exists(resume_path)
            with open(resume_path, 'r') as f:
                samples = json.load(f)
            
        sample_no = 0
        for sample in tqdm(samples):
            if resume_path and self.config.ans_key in sample:
                continue
            question, texts, images = dataset.load_sample_retrieval_data(sample)
            try:
                final_ans, final_messages = self.predict(question, texts, images)
            except RuntimeError as e:
                print(e)
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                final_ans, final_messages = None, None
            
            # Sample
            print(final_messages)

            sample[self.config.ans_key] = final_ans
            if self.config.save_message:
                sample[self.config.ans_key+"_message"] = final_messages
            torch.cuda.empty_cache()
            self.clean_messages()
            
            sample_no += 1
            if sample_no % self.config.save_freq == 0:
                path = dataset.dump_reults(samples)
                print(f"Save {sample_no} results to {path}.")
        path = dataset.dump_reults(samples)
        print(f"Save final results to {path}.")
    
    def clean_messages(self):
        for agent in self.agents:
            agent.clean_messages()