import torch

def cls_pooling(outputs, attention_mask=None):
    return outputs.last_hidden_state[:, 0, :]  # CLS token is first toke

def mean_pooling(outputs, attention_mask):
    token_embeddings = outputs[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def eos_pooling(outputs, attention_mask):
    # useful for decoder-only models like Llama, Mistral, Qwen etc.
    # Check padding, to get correct eos token. Rare, but some models use
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return outputs.last_hidden_state[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = outputs.last_hidden_state.shape[0]
        return outputs.last_hidden_state[
            torch.arange(batch_size, device=outputs.last_hidden_state.device), sequence_lengths
        ]