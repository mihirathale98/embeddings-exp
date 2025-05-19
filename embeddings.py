
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import List
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from typing import List, Tuple
import torch

def get_token_embeddings_batch(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 8
) -> Tuple[List[List[str]], torch.Tensor]:
    """Get token embeddings for a batch of texts using batching."""
    all_token_lists = []
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        with torch.no_grad():
            outputs = model(**tokens)
        
        embeddings = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        input_ids = tokens["input_ids"]
        token_lists = [tokenizer.convert_ids_to_tokens(seq) for seq in input_ids]
        
        all_token_lists.extend(token_lists)
        all_embeddings.append(embeddings)

    # Concatenate all batches along the 0th dimension (batch dimension)
    full_embeddings = torch.cat(all_embeddings, dim=0)  # (total_texts, seq_len, hidden_dim)
    
    return all_token_lists, full_embeddings



def compute_topk_token_indices(q_embeds, d_embeds, attention_mask, k=5):
    """Compute the top-k most relevant document token indices based on similarity to the question."""
    q_embeds = F.normalize(q_embeds, p=2, dim=1)
    
    doc_embeddings_list = []
    for i in range(d_embeds.size(0)):
        d_embed = d_embeds[i]  # (seq_len, hidden_dim)
        mask = attention_mask[i].bool()  # to ignore padding
        d_embed = d_embed[mask]
        d_embed = F.normalize(d_embed, p=2, dim=1)

        sim_matrix = torch.matmul(q_embeds, d_embed.T)  # (q_len, d_len)
        relevance_scores = sim_matrix.max(dim=0).values  # (d_len,)
        topk_indices = torch.topk(relevance_scores, k=min(k, relevance_scores.size(0))).indices
        print(topk_indices)

        selected_embeddings = d_embed[topk_indices]
        doc_embedding = selected_embeddings.mean(dim=0)  # (hidden_dim,)
        doc_embeddings_list.append(doc_embedding.cpu())
    
    return torch.stack(doc_embeddings_list)  # (num_docs, hidden_dim)


def build_filtered_doc_embeddings(question: str, documents: List[str], tokenizer, k=5):
    """Build question-aware document embeddings."""
    # Question embeddings
    q_tokens, q_embeds = get_token_embeddings_batch([question])
    q_embeds = q_embeds.squeeze(0)  # (q_len, hidden_dim)
    q_embeds = q_embeds[tokenizer(question, return_tensors="pt")["attention_mask"].squeeze().bool().to(device)]

    # Document embeddings
    d_tokens, d_embeds = get_token_embeddings_batch(documents)
    attention_mask = tokenizer(documents, return_tensors="pt", padding=True, truncation=True)["attention_mask"].to(device)

    doc_embeddings = compute_topk_token_indices(q_embeds, d_embeds, attention_mask, k=k)
    return doc_embeddings

