import re
class BPEWordTokenizer:
    """
    Attributes:
        vocabulary: List of subword tokens including special tokens.
        vocabulary_size : Total number of tokens in vocabulary.
        token_to_index: Mapping from tokens to indices.
        index_to_token: Mapping from indices to tokens.
        pad_token_id: Index of the padding token.
        unknown_token_id: Index of the unknown token.
        tokenized_corpus: Cached tokenized corpus after BPE training.
    """
    UNKNOWN_TOKEN = ""
    PAD_TOKEN = ""
    END_WORD = ""
    
    CLS_TOKEN = ""
    MASK_TOKEN = ""
    SEP_TOKEN = ""
    
    def __init__(
        self, 
        texts : list[str] | str,
        vocabulary: list[str] | None = None,
        num_merges : int = 100,
        ) -> None:
        """Inialize the BPEWordTokenizer

        Args:
            texts: A list of strings or a string representing the text corpus.
            vocabulary: Optional list of strings with unique tokens.
            num_merges: Defines how many rounds of merges should be performed
            when learning the BPE merges..
        """
        
        if isinstance(texts, str): #Normalize to list of strings
            texts = [texts]
            
        
        corpus = corpus.lower()
        
        words = re.split(r'([,.:;?_!"()\']|--|\s)', corpus)
        vocab = sorted(set([word.strip() for word in words if word.strip() != '']))
        
        self.special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>", "<MASK>", "<CLS>", "<SEP>"]
        
        
        vocab = self.special_tokens + vocab
        self.text_to_token_ids = {word : index for index, word in enumerate(self.vocabulary)}
        self.tokens_ids_to_text = {index : word for word, index in enumerate(self.vocabulary)}
        
        
        self.pad_token_id = self.text_to_token_ids["<pad>"]
        self.sos_token_id = self.text_to_token_ids["<sos>"]
        self.eos_token_id = self.text_to_token_ids["<eos>"]
        self.unk_token_id = self.text_to_token_ids["<unk>"]
        self.mask_token_id = self.text_to_token_ids["<MASK>"]
        self.cls_token_id = self.text_to_token_ids["<CLS>"]
        self.sep_token_id = self.text_to_token_ids["<SEP>"]
        
    def _split_text(self, text : str) -> list[str]:
        """
        Split a string into a wubword tokens using a learning BPE merges
        """
        tokens = []
        for word in text.strip().split():
            # Split the string into characters and add special END_word token "</w>"
            characters = list(word) + [self.END_WORD]
            
            # Merge individual characters according to learned BPE merges
            for pair in self.merges
    
    
    def joint_text(self, tokens : list[str]) -> str:
        """
        Join subwords tokens into full string∫
        """
        words = []
        for token in tokens:
            if token.startswith("##"):
                if words:
                    words[-1] += token[2:] # Remove ## 
        
        else:
            words.append(token)

        return " ".join(words).strip()
    
    def encoder(self, text : str) -> list[int]:
        """
        Encode a string into a list of tokens indices
        """
        tokens_ids = []
        for token in self._split_text(text=text):
            token_id = self.text_to_token_ids.get(token, self.unk_token_id)
            tokens_ids.append(token_id)
        final_tokens = [self.cls_token_id] + tokens_ids + [self.sep_token_id]
        return final_tokens
    
    def decoder(self, tokens_ids): 
        """
        Decode list of tokens IDs back to original text
        """
        if hasattr(tokens_ids, "flatten"):
            tokens_ids = tokens_ids.flatten().tolist()
        
        elif hasattr(tokens_ids, "tolist"):
            tokens_ids = tokens_ids.tolist()
        
        
        elif isinstance(tokens_ids, int):
            tokens_ids = [tokens_ids]
            
        tokens = []
        for token_id in tokens_ids:
            if token_id in [self.cls_token_id, self.sep_token_id, self.pad_token_id]:
                continue
            tokens.append(self.tokens_ids_to_text.get(token_id, self.unk_token_id)) # Usually just map to [UNK]
        
        return self.joint_text(tokens=tokens)
    
    def get_vocab_size(self):
        return len(self.text_to_token_ids)
    