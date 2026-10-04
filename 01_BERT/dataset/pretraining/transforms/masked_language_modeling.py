
from typing import List, Tuple
from dataset.tokenizer import BPEWordTokenizer


class MLM:
    
    def __init__(
        self,tokenizer: BPEWordTokenizer,mask_probability: float = 0.15, mask_replacement_probability : float = 0.80, p_replacement_random: float = 0.1) -> None:
        """Initialize the Masked Language Modeling (MLM) transform."""
        self.tokenizer = tokenizer
        self.mask_probability = mask_probability
        self.mask_replacement_probability = mask_replacement_probability
        self.p_replacement_random = p_replacement_random
        