import torch
import torch.nn as nn
import torch.nn.functional as F

def get_model(num_units=4096):
    return ActivationNet(num_units=num_units)

class ActivationNet(nn.Module):
    def __init__(self, num_units=4096):
        self.phi = nn.Sequential(
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
        )

    def forward(self, input, query, y=None, few=False, shot=10):
        W = self.phi(input) # (C, u) -> (C, u)
        logits = query.matmul(W.transpose(0, 1)) # (B, u) @ (u, C) -> (B, C)
        if not few: # y (B,)
            assert y is not None
            pred = torch.argmax(logits, 1) # (B,)
            loss = F.cross_entropy(logits, y) 
            correct_pred = pred.int() == y.int()
            acc = torch.mean(correct_pred.float())
            return loss, acc
        else:
            assert logits.shape[1] % shot == 0
            max_responses = []
            for i in range(logits.shape[1] / shot):
                max_response = torch.max(logits[:, i * shot: (i + 1) * shot], dim=1, keepdim=True) # (B, shot) -> # (B, 1)
                max_responses.append(max_response)
            logits = torch.cat(max_responses, dim=1) # (B, C / shot) i.e. (B, real_C)
            pred = torch.argmax(logits, 1) # (B,)

            if y is None:
                return pred

            loss = F.cross_entropy(logits, y)
            correct_pred = pred.int() == y.int()
            acc = torch.mean(correct_pred.float())
            return loss, acc
