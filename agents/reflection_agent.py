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
from agents.base_agent import Agent

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
        
    def get_initial_prompt(self, question: str, plan: str = None) -> str:
        """
        Generate the initial prompt for either planning or first answer generation.
        """
        if plan is None:
            # Planning prompt - focused on DocVQA
            return f"""Break down the question into key points to analyze: {question}

Required Analysis Points:
1. Main Question Components:
   - Identify core concepts and terms
   - List specific aspects to investigate
   - Note any constraints or conditions

2. Information Requirements:
   - What specific details are needed
   - Which document sections to focus on
   - Any supporting evidence required

3. Answer Structure:
   - Key points to cover
   - Required level of detail
   - Format of response

"""
        # Answer prompt - focused on DocVQA
        return f"""Generate a comprehensive technical answer based on the provided plan:

Question: {question}
Plan: {plan}

Requirements:
1. Technical Depth and Precision:
   - Employ domain-specific terminology and technical concepts
   - Reference specific methodologies, frameworks, or approaches mentioned
   - Include relevant technical metrics, measurements, or parameters
   - Explain technical relationships and dependencies
   - Use precise technical language while maintaining clarity

2. Answer Structure and Format:
   - Begin with a direct, concise response to the core question
   - Follow with 2-4 sentences of technical elaboration
   - Include specific technical details that support the answer
   - Maintain logical flow between technical concepts
   - Ensure technical accuracy while being comprehensive

3. Technical Context and Fallback:
   - If specific answer not found, state "The answer cannot be determined from the provided document. However..."
   - Provide relevant technical background and context
   - Explain related technical concepts and their relationships
   - Include technical implications or applications
   - Reference similar technical approaches or methodologies
   - Maintain technical accuracy in all provided information
"""

    def get_reflection_prompt(self, question: str, plan: str, current_ans: str) -> str:
        """
        Generate the reflection prompt for analyzing and improving the answer.
        """
        return f"""Technical review and improvement:

Question: {question}
Plan: {plan}
Current answer: {current_ans}

Improve based on:
1. Technical Accuracy:
   - Use precise domain-specific terminology
   - Ensure technical terms are used correctly
   - Maintain consistency in technical concepts
   - Verify technical relationships are accurate

2. Answer Completeness:
   - Provide a direct, specific answer to the question
   - Include only the most relevant technical details
   - Answer MUST be exactly 1-2 sentences total
   - NO paragraphs or multiple sentences allowed
   - Keep it extremely concise and focused

3. If information unavailable:
   - State "The answer cannot be determined from the provided document. However..."
   - Provide only 1 sentence of relevant technical context
   - NO additional explanation or elaboration
   - Keep it strictly to 1-2 sentences total
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
        general_agent: Agent = self.agents[0]

        current_iter = 0
        all_messages = []
        current_ans = ""  # Initialize current_ans
        plan = None

        for current_iter in range(self.config.max_reflection_iter):
            print('='*20, f'Iter:{current_iter}')
            if current_iter == 0:
                # First iteration: Planning phase
                planning_prompt = self.get_initial_prompt(question)
                plan, plan_messages = general_agent.predict(planning_prompt, with_sys_prompt=False)
                print('='*20, f'Iter:{current_iter} plan: \n', plan)
                all_messages.extend(plan_messages)
                
                # Generate initial answer based on the plan
                answer_prompt = self.get_initial_prompt(question, plan)
                current_ans, messages = general_agent.predict(answer_prompt, texts, images, with_sys_prompt=False)
                print('='*20, f'Iter:{current_iter} current_ans: \n', current_ans)
            else:
                # Reflection phase
                reflection_prompt = self.get_reflection_prompt(question, plan, current_ans)
                print('='*20, f'Iter:{current_iter} len_reflection: {len(reflection_prompt.split())} reflection_prompt: \n', reflection_prompt)
                current_ans, messages = general_agent.predict(reflection_prompt, texts, images, with_sys_prompt=False)
                print('='*20, f'Iter:{current_iter} current_ans: \n', current_ans)

            all_messages.extend(messages)
            # print('='*20, f'Iter:{current_iter} messages: \n', messages)
            self.clean_messages()

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
            # print(final_messages)

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