from agents.multi_agent_system import MultiAgentSystem
from tqdm import tqdm
import torch
import json
from mydatasets.base_dataset import BaseDataset
import os
from retrieval.conditional_retrieval import ConditionalColbertRetrieval, ConditionalColpaliRetrieval
from models.qwen import Qwen2VL
from agents.base_agent import Agent
import importlib
from functools import partial

class ReActAgent(MultiAgentSystem):
    def __init__(self, config):
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

        # Load image retriever configuration and instantiate
        image_retriever_config = config.retrieval.image_retriever
        self.image_retriever = ConditionalColpaliRetrieval(image_retriever_config)

        # Load text retriever configuration and instantiate
        text_retriever_config = config.retrieval.text_retriever
        self.text_retriever = ConditionalColbertRetrieval(text_retriever_config)
    
    def predict(self, sample, question: str, all_doc_texts: list[str], all_doc_image_paths: list[str], first_retrieved_texts, first_retrieved_images) -> tuple[str, list[dict]]:
        # Initialize with general agent
        general_agent: Agent = self.agents[0]
        
        # Create the initial system prompt with available tools and question
        # Create partial functions that bind all_doc_texts and all_doc_image_paths
        partial_retrieve_text = partial(self.retrieve_text, all_doc_texts_for_sample=all_doc_texts)
        partial_retrieve_text.__doc__ = self.retrieve_text.__doc__ # Preserve docstring for the prompt
        
        tools = [partial_retrieve_text]
        conversation_history = ""
        retry_count = 0
        current_texts, current_images = first_retrieved_texts, first_retrieved_images
        concatenated_messages = []
        
        # Initialize tracking variables
        final_answer = ""
        last_response = ""
        consecutive_no_action = 0
        max_consecutive_no_action = 2
        max_retries = 3  # Maximum number of times we'll try to get a proper answer
        max_iterations = 4  # Maximum number of iterations
        current_iteration = 0  # Track current iteration
        
        while True:
            current_iteration += 1
            self.clean_messages()
            
            # Get updated prompt based on conversation history and retry count
            system_prompt = self.get_agent_prompt(tools, question, conversation_history, retry_count)
            
            # Force final answer on last iteration
            if current_iteration > max_iterations:
                final_attempt_prompt = """
                This is the final iteration. You MUST provide an answer now based on all the information gathered so far.
                Please format your response as: 
                
                Answer: <your final answer>
                
                If you cannot find the exact information, provide the most relevant information you have found.
                """
                system_prompt = f"{system_prompt}\n\n{final_attempt_prompt}"
            
            # Get response from agent
            response, messages = general_agent.predict(system_prompt, current_texts, current_images, with_sys_prompt=False)
            concatenated_messages.extend(messages)
            print('response:', response)
            
            # Check if response contains an action
            if "Action:" in response:
                consecutive_no_action = 0  # Reset counter when action is found
                action_line = response.split("Action:")[1].split("\n")[0].strip()
                observation = ""
                
                # Remove PAUSE if it exists
                action_line = action_line.replace("PAUSE", "").strip()
                
                # Handle text retrieval
                if action_line.startswith("retrieve_text:"):
                    query = action_line.replace("retrieve_text:", "").strip()
                    current_texts = partial_retrieve_text(sample=sample, query=query)
                    observation = f"Retrieved {len(current_texts)} relevant text passages."
                    if current_texts:
                        observation += f" First passage preview: {current_texts[0][:100]}..."
                else:
                    observation = f"Unknown action: {action_line}. Please use retrieve_text."
                
                # Update conversation history with response and observation
                conversation_history = f"{conversation_history}\n\n{response}\n\nObservation: {observation}"
                
                # Get next response from agent
                response, messages = general_agent.predict(system_prompt, current_texts, current_images, with_sys_prompt=False)
                concatenated_messages.extend(messages)
                last_response = response

            # If response contains final answer, validate it
            elif "Answer:" in response:
                answer = response.split("Answer:")[1].strip()
                final_answer = answer
                print('final_answer:', final_answer)
                break

            # Handle Thought responses
            elif "Thought:" in response:
                consecutive_no_action = 0  # Reset counter for thoughts
                thought_content = response.split("Thought:")[1].strip()
                
                # Update conversation history with the thought
                conversation_history = f"{conversation_history}\n\n{response}"
                
                # Get next response from agent
                response, messages = general_agent.predict(system_prompt, current_texts, current_images, with_sys_prompt=False)
                concatenated_messages.extend(messages)
                last_response = response
            
            # Handle cases where no action or answer is found
            else:
                consecutive_no_action += 1
                last_response = response
                
                # If we've had too many responses without actions, try to guide the agent
                if consecutive_no_action >= max_consecutive_no_action:
                    if retry_count < max_retries:
                        guidance_prompt = self._get_guidance_prompt("retry")
                        conversation_history = f"{conversation_history}\n\n{guidance_prompt}"
                        response, messages = general_agent.predict(system_prompt, current_texts, current_images, with_sys_prompt=False)
                        concatenated_messages.extend(messages)
                        retry_count += 1
                        consecutive_no_action = 0
                    else:
                        # If we've exhausted all retries, try one final time with a different approach
                        guidance_prompt = self._get_guidance_prompt("final")
                        conversation_history = f"{conversation_history}\n\n{guidance_prompt}"
                        response, messages = general_agent.predict(system_prompt, current_texts, current_images, with_sys_prompt=False)
                        concatenated_messages.extend(messages)
                        
                        if "Answer:" in response:
                            final_answer = response.split("Answer:")[1].strip()
                        else:
                            final_answer = "The document does not provide specific information."
                            print(f"Warning: Agent failed to produce a structured 'Answer:' for question '{question[:50]}...'. Using default no-information message.")
                            print(conversation_history, "last_response:", response)
                        break
            
            # Force break after max iterations
            if current_iteration > max_iterations:
                if not final_answer:  # If we still don't have an answer
                    final_answer = "The document does not provide specific information."
                    print(f"Warning: Reached maximum iterations ({max_iterations}) without a valid answer. Using default no-information message.")
                break
        
        return final_answer, concatenated_messages

    def _get_guidance_prompt(self, prompt_type: str) -> str:
        """Helper method to generate guidance prompts based on type."""
        prompts = {
            "concise": """
                Your answer was too detailed. Please provide a more concise answer that:
                1. Summarizes the key factual points.
                2. Directly addresses the question based on the retrieved document content.
                3. Avoids unnecessary interpretation or elaboration.
                
                Please format your response as: Answer: <your concise, factual answer>
                """,
            "retry": """
                Based on the retrieved information, please provide a concise, factual answer to the question.
                Your answer should:
                1. Summarize the key points directly from the document content.
                2. Focus on the most important information relevant to the question.
                3. If the information is not found, explicitly state that.
                
                Please format your response as: Answer: <your answer, or state if not found>
                """,
            "final": """
                We need a final answer to the question based on the document.
                Please:
                1. Summarize the most important information found.
                2. Focus on key facts or figures that directly answer the question.
                3. If the information is not present in the retrieved content, explicitly state that.
                
                Please format your response as: Answer: <your final answer, or state if not found>
                """
        }
        return prompts.get(prompt_type, "").strip()
        
    def _validate_and_filter_indices(self, indices: list[int], max_length: int, source_name: str) -> list[int]:
        """
        Helper method to validate and filter indices, removing duplicates and out-of-bounds indices.
        
        Args:
            indices: List of indices to validate
            max_length: Maximum valid index (length of the source list)
            source_name: Name of the source list for logging purposes
            
        Returns:
            List of valid, unique indices in their original order
        """
        unique_ordered_indices = []
        if indices:
            seen_indices = set()
            for idx in indices:
                if idx not in seen_indices:
                    if 0 <= idx < max_length:
                        unique_ordered_indices.append(idx)
                        seen_indices.add(idx)
                    else:
                        print(f"Warning: {source_name} received out-of-bounds index {idx} for length {max_length}")
        
        # print(f'In {source_name}: unique_ordered_indices: {unique_ordered_indices}')
        return unique_ordered_indices

    def retrieve_text(self, sample, query: str, all_doc_texts_for_sample: list[str]) -> list[str]:
        """
        retrieve_text: <query>
        Use this action to find relevant text passages from the document based on your query.
        
        Examples:
        - retrieve_text: Find information about the model's training procedure
        - retrieve_text: Look for details about the evaluation metrics used
        - retrieve_text: Search for implementation details of the attention mechanism
        
        The query should be specific about what information you're looking for.
        Returns the most relevant text passages based on semantic similarity to your query.
        """
        top_page_indices, top_page_scores = self.text_retriever.find_sample_top_k_conditional(sample, self.config.top_k, None, query)
        # print(f'In retrieve_text: len(all_doc_texts_for_sample) : {len(all_doc_texts_for_sample)}')
        # print(f'In retrieve_text: raw top_page_indices from conditional retrieval: {top_page_indices}')
        
        valid_indices = self._validate_and_filter_indices(
            top_page_indices,
            len(all_doc_texts_for_sample),
            "retrieve_text"
        )
        return [all_doc_texts_for_sample[i] for i in valid_indices]

    def get_agent_prompt(self, tools: callable, question: str = None, conversation_history: str = None, retry_count: int = 0) -> str:
        """
        Generate an adaptive system prompt that evolves based on conversation history and retry attempts.
        
        Args:
            tools: List of available tools
            question: The current question being asked
            conversation_history: Previous conversation history (if any)
            retry_count: Number of retry attempts made so far
            
        Returns:
            An adaptive system prompt that guides the agent based on previous interactions
        """
        tools_str = "\n".join([f"{tool.__doc__} \n" for tool in tools])
        
        # Base prompt that's always included
        base_prompt = f"""
        You are a specialized Document Question Answering agent. Your primary goal is to answer questions with the HIGHEST POSSIBLE ACCURACY by diligently extracting and synthesizing information from the provided document text.

        **Required Response Format:**
        You MUST use exactly one of these three formats in your response:
        1. Action: retrieve_text: <keywords>
        2. Thought: <your reasoning>
        3. Answer: <your final answer>

        **IMPORTANT: NEVER REPEAT THE SAME KEYWORDS**
        - Each Action MUST use different keywords than previous Actions
        - If you've already used certain keywords, you MUST use different ones
        - Track your previous keywords and avoid repeating them
        - If you're unsure what keywords to use next, try related technical terms

        **Process for Answering Questions:**
        **MANDATORY FIRST STEP:**
        1. You MUST use Action: retrieve_text: <keywords> at least once to search for relevant text before attempting to answer.

        **Subsequent Steps:**
        2. Use Thought: to explain your reasoning and to analyze the retrieved text.
        3. If the initial text retrieval is insufficient, you MAY use Action: retrieve_text: <new_keywords> again with DIFFERENT keywords.
        4. Once you are confident you have sufficient information (or have determined the information is not present after thorough searching), use Answer: to provide your response.

        **Keyword Search Guidelines:**
        1. Use ONLY keywords, not full sentences
        2. NEVER repeat the same keywords in subsequent searches
        3. Try different combinations of keywords for each search
        4. Use technical terms, metrics, or specific concepts
        5. Keep keywords focused and relevant to the question

        **Answer Requirements:**
        1. Be precise and factual
        2. Include specific data points when available
        3. Use exact quotes or paraphrases from the document
        4. Avoid speculation or interpretation
        5. If information is not found, state "The document does not provide specific information"

        **Tools Available:**
        {tools_str}

        **Tool Usage Guidelines:**

        1. **Text Retrieval Tool (`retrieve_text`)**:
           - Use this tool to search for relevant information.
           - **You MUST use this tool at least once before providing an Answer.**
           - Format: Action: retrieve_text: <keywords>
           - Examples:
             * Action: retrieve_text: training procedure methodology
             * Action: retrieve_text: accuracy metrics results
             * Action: retrieve_text: model architecture implementation
           - NEVER repeat these exact keywords in subsequent searches

        **Example Workflow and Answers:**

        **Example 1: Question about a specific metric**
        Q: "What is the primary performance metric reported for the new model?"
        Action: retrieve_text: performance metric primary model
        Thought: The retrieved text mentions several metrics, but specifically highlights 'Overall F1-Score' as the main indicator of performance for the new model, reported in Table 3.
        Answer: The primary performance metric reported for the new model is an Overall F1-Score of 0.82.

        **Example 2: Question requiring a list of items**
        Q: "Which datasets were used for model pre-training?"
        Action: retrieve_text: datasets pre-training model
        Thought: The document lists several datasets in the pre-training section. I should compile these into a list.
        Answer: The model was pre-trained on the following datasets: AlphaCorpus, BetaLogs, GammaWebText.

        **Example 3: Question about a qualitative feature**
        Q: "What is the main advantage of the proposed architecture?"
        Action: retrieve_text: advantage proposed architecture design
        Thought: The text indicates that the primary benefit of the new architecture is its efficiency in handling long sequences.
        Answer: The main advantage of the proposed architecture is its improved efficiency in processing long input sequences.

        **Example 4: Question requiring comparison or multiple data points**
        Q: "How does the new algorithm compare to the baseline on key tasks?"
        Action: retrieve_text: algorithm comparison baseline tasks performance
        Thought: The paper presents a comparison table (Table 5) showing performance on Task A and Task B. I need to extract the new algorithm's scores and the baseline's scores for these tasks.
        Answer: Compared to the baseline, the new algorithm shows: Task A (New: 75.5%, Baseline: 70.2%), Task B (New: 68.9%, Baseline: 65.1%).

        **Example 5: Question where information might be absent**
        Q: "What was the exact learning rate schedule used during fine-tuning?"
        Action: retrieve_text: learning rate schedule fine-tuning details
        Thought: I have searched the document for learning rate schedule details during fine-tuning, but this specific information does not appear to be provided. The paper mentions the optimizer and initial learning rate, but not the full schedule.
        Answer: The document does not provide specific information about the exact learning rate schedule used during fine-tuning.
        """.strip()

        # Add adaptive guidance based on retry count and conversation history
        adaptive_guidance = ""
        
        if retry_count > 0:
            adaptive_guidance += f"""
            **Previous Attempt Analysis:**
            This is attempt #{retry_count + 1}. Based on previous attempts:
            """
            
            if conversation_history:
                # Extract key information from conversation history
                text_retrievals = [line for line in conversation_history.split('\n') if 'retrieve_text:' in line]
                
                if text_retrievals:
                    previous_text_keywords = [q.split('retrieve_text:')[1].strip() for q in text_retrievals[-2:]]
                    adaptive_guidance += "\n* Previous text keywords used: " + ", ".join(previous_text_keywords)
                    adaptive_guidance += "\n* DO NOT use these keywords again: " + ", ".join(previous_text_keywords)
                
                adaptive_guidance += "\n\n**Adaptive Guidance:**"
                
                if retry_count == 1:
                    adaptive_guidance += """
                    * You MUST use completely different keywords than before
                    * Try alternative technical terms and concepts
                    * Consider different aspects of the question
                    * Look for related but different terminology
                    """
                elif retry_count == 2:
                    adaptive_guidance += """
                    * Use entirely new keyword combinations
                    * Focus on different technical aspects
                    * Try broader or more specific terms
                    * Consider alternative metrics or methods
                    """
                else:
                    adaptive_guidance += """
                    * Review all previous keywords and avoid them
                    * Try completely different technical terms
                    * Consider alternative approaches to the question
                    * Look for different types of information
                    """

        # Combine base prompt with adaptive guidance
        prompt = base_prompt
        if adaptive_guidance:
            prompt += f"\n\n{adaptive_guidance}"

        if question:
            prompt = f"{prompt}\n\nCurrent Question: {question}"
        
        return prompt

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
            
            # Load full document content
            question = sample[dataset.config.question_key]
            all_content_objects = dataset.load_processed_content(sample, disable_load_image=True)
            all_doc_texts = [content.txt.replace("\n", "") for content in all_content_objects]
            all_doc_image_paths = [content.image_path for content in all_content_objects]
            question, first_retrieved_texts, first_retrieved_images = dataset.load_sample_retrieval_data(sample)
            
            try:
                final_ans, final_messages = self.predict(sample, question, all_doc_texts, all_doc_image_paths, first_retrieved_texts, first_retrieved_images)
            except RuntimeError as e:
                print(e)
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                final_ans, final_messages = None, None
            
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