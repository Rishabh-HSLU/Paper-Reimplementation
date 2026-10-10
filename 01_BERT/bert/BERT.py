import torch
import torch.nn as nn
import torch.nn.functional as F



class BERTEncoder(nn.Module):
    """
    BERT Encoder module that conmbines multiple BertEncoderLayer instances to form the blocks of the BERT architecture
    """
    def __init__(self, hidden_size: int, heads: int, intermediate_size: int, num_layers: int, dropout_prob: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [BertEncoderLayer(hidden_size, heads, intermediate_size, dropout_prob) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, attention_mask: bool = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x



class BertEncoderLayer(nn.Module):
    """
    A single BERT encoder layer that consists of a MSA + LayerNorm with residual connection, followed by a feed-forward network + LayerNorm with residual connection
    """
    def __init__(self, hidden_size: int, heads: int, intermediate_size: int, dropout_prob: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.attention_layers = BertAttentionLayer(hidden_size, heads, dropout_prob)
        self.attention_norms = nn.LayerNorm(hidden_size)

        self.feed_forward_layers = BertFeedForwardLayer(hidden_size, intermediate_size, dropout_prob)
        self.feed_forward_norms = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: bool = None) -> torch.Tensor:
        attention_output = self.attention_layers(x, attention_mask)
        attention_output = self.dropout(attention_output)  # Apply dropout to the attention output
        x = self.attention_norms(x + attention_output)  # Residual connection
        feed_forward_output = self.feed_forward_layers(x)
        x = self.feed_forward_norms(x + feed_forward_output)

        return x
    
    
class BertFeedForwardLayer(nn.Module):
    """
    The MLP feed-forward layer so the model understand the complex relationships between the tokens in the sequence
    Adds non-linearity with GELU activation function so the model can learn more complex patterns in the data
    """
    def __init__(self, hidden_size: int, intermediate_size: int, dropout_prob: float):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = nn.Dropout(dropout_prob)

        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return x
    
class BertAttentionLayer(nn.Module):
    """
    Attention layer that implements the multi-head self-attention mechanism used in BERT
    """
    def __init__(self, hidden_size: int, num_attention_heads: int, dropout_prob: float):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.dropout_prob = dropout_prob

        self.project_to_queries = nn.Linear(hidden_size, hidden_size)
        self.project_to_keys = nn.Linear(hidden_size, hidden_size)
        self.project_to_values = nn.Linear(hidden_size, hidden_size)
        self.project_to_output = nn.Linear(hidden_size, hidden_size)

    def split_to_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(tensor.size(0), tensor.size(1), self.num_attention_heads, self.head_dim).transpose(1, 2)


    def forward(self, x: torch.Tensor, attention_mask: bool = None) -> torch.Tensor:
        head_dim = self.head_dim

        query = self.project_to_queries(x)
        key = self.project_to_keys(x)
        value = self.project_to_values(x)

        queries, keys, values = self.split_to_heads(query), self.split_to_heads(key), self.split_to_heads(value)


        attention_output = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask = attention_mask,
                dropout_p = self.dropout_prob,
                is_causal=False
            )

        attention_output = attention_output.transpose(1, 2).reshape(x.size(0), x.size(1), self.hidden_size)
        output = self.project_to_output(attention_output)

        return output
