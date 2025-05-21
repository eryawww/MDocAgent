from .image_retrieval import ColpaliRetrieval
from .text_retrieval import ColbertRetrieval
import copy


class ConditionalColpaliRetrieval(ColpaliRetrieval):
    def __init__(self, config):
        super().__init__(config)
    
    def find_sample_top_k_conditional(self, sample, document_embed, top_k: int, page_id_key: str, query: str):
        # Create a deep copy of the sample to avoid modifying the original
        sample_copy = copy.deepcopy(sample)
        sample_copy[self.config.image_question_key] = query
        return super().find_sample_top_k(sample_copy, document_embed, top_k, page_id_key)
    
class ConditionalColbertRetrieval(ColbertRetrieval):
    def __init__(self, config):
        super().__init__(config)
    
    def find_sample_top_k_conditional(self, sample, top_k: int, page_id_key: str, query: str):
        # Create a deep copy of the sample to avoid modifying the original
        sample_copy = copy.deepcopy(sample)
        sample_copy[self.config.text_question_key] = query
        return super().find_sample_top_k(sample_copy, top_k, page_id_key)
