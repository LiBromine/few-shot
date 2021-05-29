import torch
import torch.nn as nn
import torch.nn.functional as F

def get_model(num_units=4096):
    return ActivationNet(num_units=num_units)

class ActivationNet(nn.Module):
    def __init__(self, num_units=4096):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(num_units, num_units),
            # nn.ReLU(),
            # nn.Linear(num_units, num_units),
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, query, y=None, few=False, shot=10):
        if not few: # y (B,)
            assert y is not None
            y = y - 1
            # input: (C, u), query: (B, u)
            W = self.phi(input) # (C, u) -> (C, u)
            logits = query.matmul(W.transpose(0, 1)) # (B, u) @ (u, C) -> (B, C)
            pred = torch.argmax(logits, 1) # (B,)
        else:
            assert input.shape[-2] == shot
            # input: (C, shot_num, u), query: (B, u), y: (B, ) or None
            num_units = input.shape[-1]
            W = self.phi(input).reshape(-1, num_units) # (C * shot, u)
            logits = query.matmul(W.transpose(0, 1)) # (B, C * shot)
            max_responses = []
            assert logits.shape[1] % shot == 0
            for i in range(logits.shape[1] // shot):
                max_response = torch.max(logits[:, i * shot: (i + 1) * shot], dim=1, keepdim=True) # (B, shot) -> # (B, 1)
                max_responses.append(max_response.values)
            print(len(max_responses))
            logits = torch.cat(max_responses, dim=1) # (B, C)
            pred = torch.argmax(logits, 1) # (B,)

            if y is None:
                return pred

        loss = self.loss(logits, y)
        correct_pred = pred.int() == y.int()
        acc = torch.mean(correct_pred.float())
        return loss, acc
