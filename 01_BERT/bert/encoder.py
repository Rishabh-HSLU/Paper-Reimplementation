import torch
import torch.nn as nn
import torch.nn.functional as F

class BertAttentionLayer(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        self.project_to_queries = nn.Linear(hidden_size, hidden_size)
        self.project_to_keys = nn.Linear(hidden_size, hidden_size)
        self.project_to_values = nn.Linear(hidden_size, hidden_size)
        self.project_to_output = nn.Linear(hidden_size, hidden_size)

    def split_to_heads(self, query, key, value) -> torch.Tensor:
        queries = torch.split(query, self.head_dim, dim=-1).transpose(-2,1)
        keys = torch.split(key, self.head_dim, dim=-1).transpose(-2,1)
        values = torch.split(value, self.head_dim, dim=-1).transpose(-2,1)

        return queries, keys, values


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_dim = self.head_dim

        query = self.project_to_queries(x)
        key = self.project_to_keys(x)
        value = self.project_to_values(x)

        queries, keys, values = self.split_to_heads(query, key, value)

        attention_outputs = []
        for q,k,v in zip(queries, keys, values):
            attention_output = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False
            )
            attention_outputs.append(attention_output)

        attention_outputs = torch.cat(attention_outputs, dim=-1)
        output = self.project_to_output(attention_outputs)

        return output

class BertFeedForwardLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)

        return x

class BertEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, heads: int, intermediate_size: int, number_of_layers: int):
        super().__init__()
        self.attention_layers = nn.ModuleList([BertAttentionLayer(hidden_size, heads) for _ in range(number_of_layers)])
        self.feed_forward_layers = nn.ModuleList([BertFeedForwardLayer(hidden_size, intermediate_size) for _ in range(number_of_layers)])
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attention_layer, feed_forward_layer in zip(self.attention_layers, self.feed_forward_layers):
            attention_output = attention_layer(x)
            x = x + attention_output  # Residual connection
            feed_forward_output = feed_forward_layer(x)
            x = x + feed_forward_output  # Residual connection
            x = F.softmax(self.projection(x))  # Optional projection layer

        return x
