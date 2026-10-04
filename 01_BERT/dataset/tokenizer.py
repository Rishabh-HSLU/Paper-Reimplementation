from collections import Counter
import re
import string
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
    UNKNOWN_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"
    END_WORD = "</w>"
    
    CLS_TOKEN = "<CLS>"
    MASK_TOKEN = "<MASK>"
    SEP_TOKEN = "<SEP>"
    
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
            
        if vocabulary is None:
            self.merges, tokenized_corpus, vocabulary_set = self.learn_bpe(texts, num_merges)
            self.tokenized_corpus = tokenized_corpus
            
            #Ensuring that the basic alphanumeric characters are always included in the vocabulary
            required_chars = set(
                string.ascii_lowercase + string.ascii_uppercase + string.digits
            )
            
            vocabulary_set.update(required_chars)
            
            self.vocabulary = (
                [self.PAD_TOKEN, self.UNKNOWN_TOKEN, self.MASK_TOKEN, self.CLS_TOKEN, self.SEP_TOKEN]
                + sorted(vocabulary_set) 
            )
        else:
            self.vocabulary = vocabulary
            self.merges = [] #When there is a vocabulary not necesary
        
        
        #Buld mapping ans det the vocabulary size
        self.vocabulary_size = len(self.vocabulary)
        self.token_to_index = {token : index for index, token in enumerate(self.vocabulary)}
        self.index_to_token = {index : token for index, token in enumerate(self.vocabulary)}
        self.pad_token_id = self.token_to_index[self.PAD_TOKEN]
        self.unknown_token_id = self.token_to_index[self.UNKNOWN_TOKEN]
        self.masked_token_id = self.token_to_index[self.MASK_TOKEN]
        self.cls_token_id = self.token_to_index[self.CLS_TOKEN]
        self.sep_token_id = self.token_to_index[self.SEP_TOKEN]

    def _split_text(self, text : str) -> list[str]:
        """
        Split a string into a subword tokens using a learning BPE merges
        """
        tokens = []
        for word in text.strip().split():
            # Split the string into characters and add special END_word token "</w>"
            characters = list(word) + [self.END_WORD]
            
            # Merge individual characters according to learned BPE merges
            for pair in self.merges:
                characters = self.merge_pair_in_words(characters, pair)
            tokens.extend(characters)
        return tokens
    
    def join_text(self, tokens : list[str]) -> str:
        #Join subwords tokens into full string
        words = []
        current_word = []
        for token in tokens:
            if token.endswith(self.END_WORD): #Check whether token endws with a word boundary marker
                current_word.append(token.replace(self.END_WORD, "")) #Remove the end of word marker
                words.append("".join(current_word)) #Join the current word
                current_word = [] #Reset the current word
            else:
                current_word.append(token)
        
        if current_word: #If there are any remaining tokens that do not end with the end of word marker, join them as a single word
            words.append("".join(current_word))
        return " ".join(words).strip()
    
    def encoder(self, text : str) -> list[int]:
        """    
        Encode a string into a list of tokens indices
        """
        tokens_ids = []
        for token in self._split_text(text=text):
            token_id = self.token_to_index.get(token, self.unknown_token_id)
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
            tokens.append(self.index_to_token.get(token_id, self.unknown_token_id)) # Usually just map to [UNK]
        
        return self.join_text(tokens=tokens)
    
    def get_vocab_size(self):
        return len(self.token_to_index)
    
    def bpe_initialize(self, dataset : list[str]):
        """
        Implements the initialization step of the byte pair encoding algorithm
        """
        corpus = []
        vocabulary = {self.END_WORD}
        for paragrahp in dataset:
            for word in paragrahp.split(" "):
                chars = list(word) #Converting the words in chars
                chars.insert(len(word), self.END_WORD) #Adding the END of SYMBOL TOKEN
                corpus.append(chars)
                
                vocabulary.update(chars)
        
        return corpus, vocabulary
        
    def get_pair_frequencies(self, corpus : list[list[str]]) -> Counter[tuple[str,str], int]:
        """
        Calculates the frequency of adjacent character pairs in a corpus
        """
        pairs = Counter()
        for word in corpus:
            #We are looping through the position of every token but the last one!
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1]) #We are creating a tuple representing the adjacent pair of consisiting of the i-th and (i+1)-th token in the word
                pairs[pair] +=1
        return pairs
        
    
    def merge_pair_in_words(self, word : list[str], pair_to_merge : tuple[str, str]) -> list[str]:
        """
        Merges adjacent occurrences of a specfied pair of characters in a word
        """
        merged_symbol = pair_to_merge[0] + pair_to_merge[1]
        index = 0
        new_word = []
        while index < len(word):
            
            if index < len(word) - 1 and (word[index], word[index + 1]) == pair_to_merge:#If this osition and the next match, merge them
                new_word.append(merged_symbol)
                index += 2
            else:
                new_word.append(word[index])
                index += 1
        return new_word
    
    def learn_bpe(self, dataset : list[str], num_merges : int):
        """
        Learns byte pair encoding (BPE) merge operations from a dataset
        """
        corpus, vocabulary = self.bpe_initialize(dataset=dataset) # Step 1: Initialize corpus as a list of lust of characters
        
        merges = []
        for _  in range(num_merges):
            pair_freqs = self.get_pair_frequencies(corpus) #Step 2: Count all the pair frequencies
            if not pair_freqs:
                break
            
            most_freq_pair, freq = pair_freqs.most_common(1)[0] #Step 3: Pick the most frequent pair
            if freq < 1:
                break #If there isnt any pairs to merge
            merges.append(most_freq_pair)
            new_token = most_freq_pair[0] + most_freq_pair[1]
            vocabulary.add(new_token)
            
            #Step 4: Merge that pair in every word of the corpus
            new_corpus = []
            for word in corpus:
                new_word = self.merge_pair_in_words(word, most_freq_pair)
                new_corpus.append(new_word)
            corpus = new_corpus
        
        #Returning the list of merges, the vocabulary, and the final tokenized corpus
        return merges, vocabulary, corpus
        
   
    
