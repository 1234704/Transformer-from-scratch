import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    """
    def __init__(self, embedding_dim=512, heads=8):
        super(MultiHeadAttention, self).__init__()
       
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.head = self.embedding_dim // self.heads
        self.query = nn.Linear(self.head, self.head)
        self.key = nn.Linear(self.head, self.head)
        self.value = nn.Linear(self.head, self.head) 
        
        self.fc_out = nn.Linear(self.head * self.heads, embedding_dim)
    def forward(self, query, key, value):
        """
        query est de dim : [batch_size, seq_len, embed_dim = heads*head]
        """
        batch_size = query.size(0)
        query_len, key_len, value_len = query.size(1), key.size(1), value.size(1)
        query = query.reshape(batch_size, query_len, self.heads, self.head)
        key = key.reshape(batch_size, key_len, self.heads, self.head)
        value = value.reshape(batch_size, value_len, self.heads, self.head)
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        QK_transpose = torch.einsum("bqhd,bkhd->bhqk", [query, key]) # batch_size,seq_len,heads,dim_embedding----> batch_size,heads,seq_len,seq_len
        scores = F.softmax(QK_transpose / math.sqrt(self.head))
        # Multiply attention scores by the values
        output = torch.einsum("nhql,nlhd->nqhd", [scores, value]).reshape(
            batch_size, query_len, self.heads * self.head
        )

        output = self.fc_out(output)  # Final fully connected layer
        return output



        

        