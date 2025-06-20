import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=1, num_objects=4):
        super().__init__()
        total_in_dim = in_dim * num_objects
        self.mlp = nn.Sequential(
            nn.Linear(total_in_dim, total_in_dim),
            nn.BatchNorm1d(total_in_dim),
            nn.ReLU(),
            nn.Linear(total_in_dim, out_dim),
            # nn.BatchNorm1d(out_dim)  
        )
    def forward(self, inputs):
        fused = torch.cat(inputs, dim=-1)
        out = self.mlp(fused)
        return out


# class AddSubFusion(nn.Module):
#     def __init__(self, in_dim, out_dim, num_objects=4):
#         super().__init__()
#         num_pairwise = num_objects * (num_objects - 1)
#         self.mlp = nn.Sequential(
#             nn.Linear(in_dim * (1 + num_pairwise), out_dim),
#             nn.BatchNorm1d(out_dim),
#             nn.ReLU(),
#             nn.Linear(out_dim, out_dim),
#             nn.BatchNorm1d(out_dim)  # Optional second normalization
#         )

#     def forward(self, inputs):
#         sum_vec = torch.stack(inputs, dim=0).sum(dim=0)
#         diffs = []
#         for i in range(len(inputs)):
#             for j in range(len(inputs)):
#                 if i != j:
#                     diffs.append(inputs[i] - inputs[j])
#         fused = torch.cat([sum_vec] + diffs, dim=-1)
#         return self.mlp(fused)


# class GatedFusion(nn.Module):
#     def __init__(self, in_dim):
#         super().__init__()
#         self.gate = nn.Sequential(
#             nn.Linear(in_dim * 2, in_dim),
#             nn.Sigmoid()
#         )
#         self.linear = nn.Sequential(
#             nn.Linear(in_dim * 2, in_dim),
#             nn.BatchNorm1d(in_dim)  # Added normalization after fusion
#         )

#     def forward(self, inputs):
#         fused = inputs[0]
#         for i in range(1, len(inputs)):
#             x2 = inputs[i]
#             concat = torch.cat([fused, x2], dim=-1)
#             g = self.gate(concat)
#             fused = g * fused + (1 - g) * x2
#         return self.linear(torch.cat([fused, concat], dim=-1))


class Transformer(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=1, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)  # Optional post-transformer normalization
        )

    def forward(self, inputs):
        tokens = torch.stack(inputs, dim=1)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.pool(pooled)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, 1, dim))  # Learnable query token
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.output_head = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)  # Predict a scalar (e.g. affinity)
        )

    def forward(self, inputs):
        """
        inputs: list of [batch_size, dim] tensors representing fused representations,
                e.g., [ligand_vec, pocket_vec, protein_vec, surface_vec]
        """
        context = torch.stack(inputs, dim=1)  # [B, N, D]
        query = self.query_token.expand(context.size(0), -1, -1)  # [B, 1, D]
        attended, _ = self.cross_attn(query, context, context)  # [B, 1, D]
        return self.output_head(attended.squeeze(1))  # [B, 1]