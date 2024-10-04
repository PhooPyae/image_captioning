import torch
from torch import nn
from torch.nn import functional as F
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        matrix = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)
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
        self.heads = nn.ModuleList([Head(n_embedding, head_size, max_sequence) for _ in range(n_head)])
        self.proj = nn.Linear(n_embedding, n_embedding)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        head_outputs = torch.cat([head(x) for head in self.heads], dim=-1)
        proj_output = self.proj(head_outputs)
        return self.dropout(proj_output)

class FeedForward(nn.Module):
    def __init__(self, n_embedding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedding, 4 * n_embedding),  # Expanding
            nn.ReLU(),
            nn.Linear(4 * n_embedding, n_embedding),  # Projecting back to original size
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.net(x)

class Blocks(nn.Module):
    def __init__(self, n_embedding, n_head, max_sequence):
        super().__init__()
        self.attention = MultiHeadAttention(n_embedding, n_head, max_sequence)
        self.feed_forward = FeedForward(n_embedding)
        self.layernorm1 = nn.LayerNorm(n_embedding)
        self.layernorm2 = nn.LayerNorm(n_embedding)

    def forward(self, x):
        x = x + self.attention(self.layernorm1(x))
        x = x + self.feed_forward(self.layernorm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, token_size, n_embedding, max_sequence, n_layers, n_head):
        super().__init__()
        self.token_embedding = nn.Embedding(token_size, n_embedding)
        self.positional_embedding = nn.Embedding(max_sequence, n_embedding)
        self.blocks = nn.Sequential(*[Blocks(n_embedding, n_head, max_sequence) for _ in range(n_layers)])
        self.layernorm = nn.LayerNorm(n_embedding)
        self.output_head = nn.Linear(n_embedding, token_size)

    def forward(self, x_input, target=None):
        B, T = x_input.shape
        token_emb = self.token_embedding(x_input)  # (B, T, C)
        pos_emb = self.positional_embedding(torch.arange(T, device=x_input.device))  # (T, C)
        x = token_emb + pos_emb  # Add positional embedding

        x = self.blocks(x)  # Pass through the transformer layers
        x = self.layernorm(x)
        logits = self.output_head(x)  # Final projection

        if target is None:
            return logits, None  # During inference, we return only the logits

        # Compute cross-entropy loss during training
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        target = target.view(B * T)
        loss = F.cross_entropy(logits, target)
        return logits, loss