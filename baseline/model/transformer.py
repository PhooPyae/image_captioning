import torch
from torch import nn as nn
from torch.nn import functional as F
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''
Rewrite this transformer from scratch
'''
class Head(nn.Module):
    def __init__(self, n_embedding, head_size, max_sequence):
        super().__init__()
        self.key = nn.Linear(n_embedding, head_size, bias=False)
        self.query = nn.Linear(n_embedding, head_size, bias=False)
        self.value = nn.Linear(n_embedding, head_size, bias=False)
        self.dropout = nn.Dropout(0.2)
        self.register_buffer('tril', torch.tril(torch.ones(max_sequence, max_sequence)))
        
    def forward(self, x):
        logger.debug(f'{x.shape=}')
        B, T, C = x.shape
        
        k = self.key(x)
        q = self.query(x)
        matrix = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        logger.debug(f'{matrix.shape=}')
        masked_matrix = matrix.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        after_softmax = F.softmax(masked_matrix, dim=-1)
        after_softmax = self.dropout(after_softmax)

        v = self.value(x)
        output = after_softmax @ v
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embedding, n_head, max_sequence):
        super().__init__()
        head_size = n_embedding // n_head
        self.multiheads = nn.ModuleList([Head(n_embedding, head_size, max_sequence) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embedding)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        multihead_output = torch.concat([h(x) for h in self.multiheads], dim=-1)
        logger.debug(f'{multihead_output.shape=}')
        proj_output = self.proj(multihead_output)
        logger.debug(f'{proj_output.shape=}')
        output = self.dropout(proj_output)
        return output

class FeedForward(nn.Module):
    def __init__(self, n_embedding):
        super().__init__()
        
        self.ffn = nn.Sequential(
            nn.Linear(n_embedding, 5 * n_embedding),
            nn.ReLU(),
            nn.Linear(5 * n_embedding, n_embedding),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        return self.ffn(x)
        
class Blocks(nn.Module):
    def __init__(self, n_embedding, n_head, max_sequence):
        super().__init__()
        
        self.multihead_att = MultiHeadAttention(n_embedding, n_head, max_sequence)
        self.layernorm_1 = nn.LayerNorm(n_embedding)
        self.layernorm_2 = nn.LayerNorm(n_embedding)
        self.ffn = FeedForward(n_embedding)
    
    def forward(self, x):
        x = self.layernorm_1(x)
        x = x + self.multihead_att(x)
        x = self.layernorm_2(x)
        x = x + self.ffn(x)
        return x
        
class Transformer(nn.Module):
    '''
    Input -> Tokenize -> Embedding vector -> positional encoding
    
    '''
    def __init__(self, token_size, n_embedding, max_sequence, n_layers, n_head):
        super().__init__()
        
        self.embedding = nn.Embedding(token_size, n_embedding)
        self.positional_embedding = nn.Embedding(max_sequence, n_embedding)
        self.blocks = nn.Sequential(*[Blocks(n_embedding, n_head, max_sequence) for _ in range(n_layers)])
        self.layernorm_final = nn.LayerNorm(n_embedding)
        self.output_head = nn.Linear(n_embedding, token_size)
    
    def forward(self, x_input, target):
        logger.debug(f'{x_input.shape=}')
        logger.debug(f'{target.shape=}')
        B, T = x_input.shape
        
        embd = self.embedding(x_input)
        pos_embed = self.positional_embedding(torch.arange(T))
        x = embd + pos_embed
        logger.debug('----------------------')

        x = self.blocks(x)
        logger.debug('----------------------')
        x = self.layernorm_final(x)
        output = self.output_head(x)
        logger.debug(f'{output.shape=}')
        if target is None:
            loss = None
        else:
            logger.debug(f'Get Loss')
            B, T, C = output.shape
            output = output.view(B*T, C)
            target = target.view(B*T)
            logger.debug(f'{output.shape=} - {target.shape=}')
            loss = F.cross_entropy(output, target)
            logger.debug(f'{loss=}')
        
        return output, loss