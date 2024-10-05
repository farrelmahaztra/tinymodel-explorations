from tinymodel import TinyModel, tokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import torch

lm = TinyModel()

def get_token_activations(prompt, layer, component):
    tok_ids = tokenizer.encode(prompt)
    tok_ids_tensor = torch.tensor([tok_ids])
    key = f"{component}{layer}"
    with torch.no_grad():
        activations = lm[key](tok_ids_tensor)
    
    if activations.dim() > 2:
        activations = activations.squeeze(0)
    
    return activations.detach(), tok_ids

def visualize_activations(prompt, layer, component, top_k):
    activations, tok_ids = get_token_activations(prompt, layer, component)
    tokens = []
    for tid in tok_ids:
        token = tokenizer.decode(tid)
        if isinstance(token, list):
            tokens.extend(token)
        else:
            tokens.append(token)
    
    _, feature_indices = torch.sort(activations.max(dim=0)[0], descending=True)
    top_features = feature_indices[:top_k].tolist()
    
    feature_activations = activations[:, top_features].numpy()
    
    plt.figure(figsize=(20, 10))
    sns.heatmap(feature_activations.T, annot=True, fmt=".2f", cmap="YlOrRd", 
                xticklabels=tokens, yticklabels=[f"Feature {f}" for f in top_features])
    plt.title(f"Top {top_k} Feature Activations")
    plt.xlabel("Tokens")
    plt.ylabel("Features")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('activations.png')
    

prompt = "There once was a knight in shining armor"

# gen = lm.generate(prompt)
# print(gen)

visualize_activations(prompt, layer=2, component='M', top_k=10)
