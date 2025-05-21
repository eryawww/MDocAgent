from agents.multi_agent_system import MultiAgentSystem
from tqdm import tqdm
import torch
import json
from mydatasets.base_dataset import BaseDataset
import hydra
import os
from retrieval.image_retrieval import ColpaliRetrieval
from retrieval.text_retrieval import ColbertRetrieval
from models.qwen import Qwen2VL
class ReActAgent(MultiAgentSystem):
    def __init__(self, config):
        super().__init__(config)

        # Load image retriever configuration and instantiate
        image_retriever_config = config.retrieval.image_retriever
        self.image_retriever = ColpaliRetrieval(image_retriever_config)

        # Load text retriever configuration and instantiate
        text_retriever_config = config.retrieval.text_retriever
        self.text_retriever = ColbertRetrieval(text_retriever_config)
    
    def predict(self, question: str, texts: list[str], images: list[str]) -> tuple[str, list[dict]]:
        # Initialize with general agent
        general_agent: Qwen2VL = self.agents[0]
        
        # Create the initial system prompt with available tools and question
        tools = [self.retrieve_text, self.retrieve_image]
        system_prompt = self.get_agent_prompt(tools, question)
        
        conversation_history = system_prompt
        current_texts, current_images = [], []
        concatenated_messages = []
        
        # Initialize default final answer
        final_answer = ""
        
        # Get initial response from agent
        response, messages = general_agent.predict(conversation_history, current_texts, current_images)
        concatenated_messages.extend(messages)
        
        iteration = 0
        while iteration < self.config.max_iterations:
            # Check if response contains an action
            if "Action:" in response:
                action_line = response.split("Action:")[1].split("\n")[0].strip()
                observation = ""
                
                # Remove PAUSE if it exists
                action_line = action_line.replace("PAUSE", "").strip()
                
                # Handle text retrieval
                if action_line.startswith("retrieve_text:"):
                    query = action_line.replace("retrieve_text:", "").strip()
                    current_texts = self.retrieve_text(query, texts)
                    observation = f"Retrieved {len(current_texts)} relevant text passages."
                    # Add more detailed observation if texts were retrieved
                    if current_texts:
                        observation += f" First passage preview: {current_texts[0][:100]}..."
                
                # Handle image retrieval
                elif action_line.startswith("retrieve_image:"):
                    query = action_line.replace("retrieve_image:", "").strip()
                    current_images = self.retrieve_image(query, images)
                    observation = f"Retrieved {len(current_images)} relevant images."
                else:
                    # Handle unknown action
                    observation = f"Unknown action: {action_line}. Please use retrieve_text or retrieve_image."
                
                # Update conversation history with response and observation
                conversation_history = f"{conversation_history}\n\n{response}\n\nObservation: {observation}"
                
                # Get next response from agent with updated conversation history and latest retrieved content
                response, messages = general_agent.predict(conversation_history, current_texts, current_images)
                
                concatenated_messages.extend(messages)
                final_answer = response
            # If response contains final answer, break the loop
            elif "Answer:" in response:
                final_answer = response.split("Answer:")[1].strip()
                break
            
            # If no action or answer found, break to avoid infinite loop
            else:
                final_answer = response
                break
                
            iteration += 1
        
        return final_answer, concatenated_messages
        
    def retrieve_image(self, query: str, images: list[str]) -> list[str]:
        """
        retrieve_image: <query>
        Use this action to find relevant images from the document based on your query.
        
        Examples:
        - retrieve_image: Find any diagrams showing the model architecture
        - retrieve_image: Look for performance comparison charts
        - retrieve_image: Search for visualizations of the training process
        
        The query should be specific about what visual information you're looking for.
        Returns the most relevant images based on semantic similarity to your query.
        """
        top_page_indices, top_page_scores = self.image_retriever.find_sample_top_k(query, self.config.top_k, self.config.r_image_key)
        return [images[i] for i in top_page_indices[:self.config.top_k]] if top_page_indices else images
        
    def retrieve_text(self, query: str, texts: list[str]) -> list[str]:
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
        top_page_indices, top_page_scores = self.text_retriever.find_sample_top_k(query, self.config.top_k, self.config.r_text_key)
        return [texts[i] for i in top_page_indices[:self.config.top_k]] if top_page_indices else texts

    def get_agent_prompt(self, tools: callable, question: str = None):
        tools_str = "\n".join([f"{tool.__doc__} \n" for tool in tools])
        
        prompt = f"""
        You are a technical document question answering agent that operates in a loop of Thought, Action, PAUSE, Observation.
        Your goal is to provide precise, technically accurate answers by actively retrieving and analyzing relevant information from the document.

        Process Flow:
        1. Thought Phase:
           - Analyze the technical requirements of the question
           - Identify key technical concepts and components
           - Determine what specific information needs to be retrieved
           - Plan the technical approach for answering

        2. Action Phase:
           Use one of these technical retrieval actions:
           {tools_str}
           Then return PAUSE.

        3. Observation Phase:
           - Analyze the retrieved technical content
           - Evaluate the relevance and completeness
           - Identify any technical gaps or uncertainties
           - Determine if additional retrieval is needed

        4. Answer Phase:
           Provide a technically precise answer that:
           - Uses proper technical terminology
           - Maintains technical accuracy
           - Follows logical structure
           - Cites specific technical details from the document

        Example Technical Session:

        Question: What is the model architecture described in the paper?
        Thought: I need to retrieve technical details about the model's architecture. This likely involves both text descriptions and architectural diagrams.
        Action: retrieve_text: model architecture neural network structure
        PAUSE

        Observation: Retrieved technical content about the model architecture, including layer specifications and connectivity patterns.

        Thought: I should also check for any architectural diagrams that might provide visual details.
        Action: retrieve_image: model architecture diagram neural network
        PAUSE

        Observation: Retrieved architectural diagram showing the model's layer structure and connections.

        Answer: The paper describes a transformer-based architecture with 12 encoder layers, each containing multi-head self-attention (8 heads) and feed-forward networks (2048 hidden units). The model uses layer normalization and residual connections, as shown in Figure 2. The input embeddings are 768-dimensional, and the architecture includes a special [CLS] token for classification tasks.

        Remember to:
        1. Use precise technical terminology
        2. Maintain technical accuracy
        3. Structure your answer logically
        4. Cite specific technical details from the document
        5. Acknowledge any technical uncertainties
        """.strip()

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