
from typing import List, Tuple
from dataset.tokenizer import BPEWordTokenizer
import torch

class MaskedLanguageModeling:
    """
    Takes a tokenized dataset and applies maskked language modeling
    """
    def __init__(
        self,tokenizer: BPEWordTokenizer,
        context_length: int = 128,
        batch_size : int = 32,
        p_mask: float = 0.15, #Base probabilty of selecting a token for the MLM task
        p_replacement_mask : float = 0.80, #The probabilty of replacing a selected token with a [MASK] token
        p_replacement_random: float = 0.10 #The probability of replacing a token with a radom token
        ) -> None:
        """Initialize the Masked Language Modeling (MLM) transform."""
        
        assert 0.0 <= p_mask <= 1.0, "The mask probability must be between 0 and 1"
        assert 0.0 <= p_replacement_mask <= 1.0, "The mask replacement probability must be between 0 and 1"
        assert 0.0 <= p_replacement_random <= 1.0, "The random replacement probability must be between 0 and 1"
        assert p_mask + p_replacement_mask <= 1.0, "The sum of mask probability and mask replacement probability must be less than or equal to 1"
        
        p_replancement_unchanged = 1.0 - p_replacement_mask - p_replacement_random  #The probability that the token remains unchanged
        
        
        self.tokenizer = tokenizer
        self.context_length = context_length 
        self.batch_size = batch_size
        self.p_mask = p_mask
        self.p_replacement_mask = p_replacement_mask
        self.p_replacement_random = p_replacement_random
        self.p_replacement_unchanged = p_replancement_unchanged

    def mask_function(self, batch : dict):
            # returns the input_ids and the labkes and masked_token_mask for the batch
            
        batch_update = dict()
        batch_update['labels'] = batch['inputs_ids'].clone()  # Create a copy of the input_ids to serve as labels
            
        """
            if mask == 0 -> we dont mask the token
            elif mask == 1 -> we put [MASK]
            elif mask == 2 -> we put a random replace token
            else -> we keep the token unchanged
        """
        p_mask_replace = self.p_mask * self.p_replacement_mask #the probability of putting a MASK token, this means % of the tokens will be masked
        p_random_replace = self.p_mask * self.p_replacement_random  #the probability of putting a random replacement token, this means % of the tokens will be replaced with a random token
        p_unchanged = self.p_mask * self.p_replacement_unchanged #the probability of keeping the token unchanged, this means % of the tokens will be kept unchanged
        
        random_values = torch.rand(size=batch['input_ids'].size())  # Create a matrix mask with the random values,¡
        
        mask_replace = (random_values < p_mask_replace) #if the value is below the p_mask_replace, we will put a [MASK] token, it catches everything from 0.00 to 0.12
        
        random_replace = (random_values > p_mask_replace) & (random_values < (p_mask_replace + p_random_replace)) # Create a matrix mask with the random values, if the value is between p_mask_replace and p_mask_replace + p_random_replace, we will put a random token, from 0.12 to 0.135
        
        unchanged = (random_values > p_mask_replace + p_random_replace ) & (random_values < (p_mask_replace + p_random_replace + p_unchanged)) # Create a matrix mask with the random values, if the value is between p_mask_replace + p_random_replace and p_mask_replace + p_random_replace + p_unchanged, we will keep the token unchanged, 0.135 to 0.15
        
        special_tokens = batch["special_tokens_mask"] == 1
        
        mask = torch.where(mask_replace, 1, 0) # If 1 -> MASK
        mask = torch.where(random_replace, 2 , mask)# Elif 2 random token
        mask = torch.where(unchanged, 3, mask) #Elif 3 -> Unchanged
        mask = torch.where(special_tokens, 0, mask) #Elif 0 -> nothing
        
        batch_update["masked_token_mask"] = (
            (mask == 1) | (mask == 2) | (mask == 3)
        ).bool() #1 -> MASK, 2 -> random token, 3 -> unchanged
        
        #Replace values according to mask
        input_ids = batch["input_ids"]
        
        
        #Applaying the logic of the MLM
        input_ids[mask == 1] = self.tokenizer.masked_token_id
        
        random_token = torch.randint_like(input_ids, self.tokenizer.get_vocab_size())
        input_ids[mask == 2] = random_token[mask == 2]
        
        batch_update["input_ids"] = input_ids
        #Applying so the CrossEntropyIgnore all the data that is not masked
        batch_update['labels'][~batch_update["masked_token_mask"]] = -100
        
        return batch_update