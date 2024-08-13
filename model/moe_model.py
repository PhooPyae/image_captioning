import torch
import torch.nn as nn
import torch.nn.functional as F

class LLaMAMLP(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x)
        return self.dropout(self.proj(x))
    
class LLaMAMoE(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList(LLaMAMLP(config) for _ in range(config.n_expert))

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derived from: https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
        See also figure 1 in https://arxiv.org/abs/2211.15841
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        x = x.view(-1, C)  # (B*T, C)
        router = self.gate(x)  # (B*T, n_expert)
        probs, indices = torch.topk(router, self.config.n_expert_per_token)  # (B*T, n_expert_per_token)
        probs = probs.softmax(dim=1, dtype=torch.float).to(dtype=x.dtype)
        masks = indices.unsqueeze(-1) == torch.arange(self.config.n_expert, device=x.device)
        masks = masks.permute(2, 0, 1)  # (n_expert, B*T, n_expert_per_token)
        y = torch.zeros_like(x)  # (B*T, C)
        for mask, expert in zip(masks, self.experts):
            token_idx, expert_idx = torch.where(mask)
            y[token_idx] += probs[token_idx, expert_idx, None] * expert(x[token_idx])
        return y.view(B, T, C)
    
class VisionEncoderDecoderMoE():
    def __init__(self, model, config: dict) -> None:
        self.model = model
        self.intermediate_size = config.intermediate_size
        self.moe_layers = nn.ModuleList(LLaMAMoE(config) for _ in range(model.config.decoder.n_layer))
        self.update_weight()
        self.add_custom_layer()

    def update_weight(self):
        for i in range(self.model.config.decoder.n_layer):
            for j in range(len(self.moe_layers[0].experts)):
                # Accessing the source weights from temp_decoder
                source_weights =self. model.decoder.transformer.h[i].mlp.c_fc.weight

                # Reshaping the weights to the desired shape
                start = 0 + (j * self.intermediate_size)
                end = self.intermediate_size * (j + 1)
                
                reshaped_weights = source_weights.view(3072, -1)[start:end,]

                # Copying the reshaped weights to the target MoE layer
                self.moe_layers[i].experts[j].fc.weight.data = reshaped_weights.clone()
                
                proj_weights = self.model.decoder.transformer.h[i].mlp.c_proj.weight
                
                reshaped_proj_weights = proj_weights.view(768, -1)[:, start:end]
                self.moe_layers[i].experts[j].proj.weight.data = reshaped_proj_weights.clone()
                
    def add_custom_layer(self):
        for i in range(len(self.model.decoder.transformer.h)):
            self.model.decoder.transformer.h[i].mlp = self.moe_layers[i]