import numpy as np
import torch
import pandas as pd
'''Code from https://github.com/Mamiglia/challenge'''

def mrr(pred_indices: np.ndarray, gt_indices: np.ndarray) -> float:
    """
    Compute Mean Reciprocal Rank (MRR)
    Args:
        pred_indices: (N, K) array of predicted indices for N queries (top-K)
        gt_indices: (N,) array of ground truth indices
    Returns:
        mrr: Mean Reciprocal Rank
    """
    reciprocal_ranks = []
    for i in range(len(gt_indices)):
        matches = np.where(pred_indices[i] == gt_indices[i])[0]
        if matches.size > 0:
            reciprocal_ranks.append(1.0 / (matches[0] + 1))
        else:
            reciprocal_ranks.append(0.0)
    return np.mean(reciprocal_ranks)


def recall_at_k(pred_indices: np.ndarray, gt_indices: np.ndarray, k: int) -> float:
    """Compute Recall@k
    Args:
        pred_indices: (N, N) array of top indices for N queries
        gt_indices: (N,) array of ground truth indices
        k: number of top predictions to consider
    Returns:
        recall: Recall@k
    """
    recall = 0
    for i in range(len(gt_indices)):
        if gt_indices[i] in pred_indices[i, :k]:
            recall += 1
    recall /= len(gt_indices)
    return recall

import numpy as np

def ndcg(pred_indices: np.ndarray, gt_indices: np.ndarray, k: int = 100) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@k)
    Args:
        pred_indices: (N, K) array of predicted indices for N queries
        gt_indices: (N,) array of ground truth indices
        k: number of top predictions to consider
    Returns:
        ndcg: NDCG@k
    """
    ndcg_total = 0.0
    for i in range(len(gt_indices)):
        matches = np.where(pred_indices[i, :k] == gt_indices[i])[0]
        if matches.size > 0:
            rank = matches[0] + 1
            ndcg_total += 1.0 / np.log2(rank + 1)  # DCG (IDCG = 1)
    return ndcg_total / len(gt_indices)



@torch.inference_mode()
def evaluate_retrieval(translated_embd, image_embd, gt_indices, max_indices = 99, batch_size=100):
    """Evaluate retrieval performance using cosine similarity
    Args:
        translated_embd: (N_captions, D) translated caption embeddings
        image_embd: (N_images, D) image embeddings
        gt_indices: (N_captions,) ground truth image indices for each caption
        max_indices: number of top predictions to consider
    Returns:
        results: dict of evaluation metrics
    
    """
    # Compute similarity matrix
    if isinstance(translated_embd, np.ndarray):
        translated_embd = torch.from_numpy(translated_embd).float()
    if isinstance(image_embd, np.ndarray):
        image_embd = torch.from_numpy(image_embd).float()
    
    n_queries = translated_embd.shape[0]
    device = translated_embd.device
    
    # Prepare containers for the fragments to be reassembled
    all_sorted_indices = []
    l2_distances = []
    
    # Process in batches - the narrow gate approach
    for start_idx in range(0, n_queries, batch_size):
        batch_slice = slice(start_idx, min(start_idx + batch_size, n_queries))
        batch_translated = translated_embd[batch_slice]
        batch_img_embd = image_embd[batch_slice]
        
        # Compute similarity only for this batch
        batch_similarity = batch_translated @ batch_img_embd.T

        # Get top-k predictions for this batch
        batch_indices = batch_similarity.topk(k=max_indices, dim=1, sorted=True).indices.numpy()
        all_sorted_indices.append(gt_indices[batch_slice][batch_indices])

        # Compute L2 distance for this batch
        batch_gt = gt_indices[batch_slice]
        batch_gt_embeddings = image_embd[batch_gt]
        batch_l2 = (batch_translated - batch_gt_embeddings).norm(dim=1)
        l2_distances.append(batch_l2)
    
    # Reassemble the fragments
    sorted_indices = np.concatenate(all_sorted_indices, axis=0)
    
    # Apply the sacred metrics to the whole
    metrics = {
        'mrr': mrr,
        'ndcg': ndcg,
        'recall_at_1': lambda preds, gt: recall_at_k(preds, gt, 1),
        'recall_at_3': lambda preds, gt: recall_at_k(preds, gt, 3),
        'recall_at_5': lambda preds, gt: recall_at_k(preds, gt, 5),
        'recall_at_10': lambda preds, gt: recall_at_k(preds, gt, 10),
        'recall_at_50': lambda preds, gt: recall_at_k(preds, gt, 50),
    }
    
    results = {
        name: func(sorted_indices, gt_indices)
        for name, func in metrics.items()
    }
    
    l2_dist = torch.cat(l2_distances, dim=0).mean().item()
    results['l2_dist'] = l2_dist
    
    return results

def generate_submission(sample_ids, translated_embeddings, output_file="submission.csv"):
    """
    Generate a submission.csv file from translated embeddings.
    """
    print("Generating submission file...")

    if isinstance(translated_embeddings, torch.Tensor):
        translated_embeddings = translated_embeddings.cpu().numpy()

    # Create a DataFrame with sample_id and embeddings

    df_submission = pd.DataFrame({'id': sample_ids, 'embedding': translated_embeddings.tolist()})

    df_submission.to_csv(output_file, index=False, float_format='%.17g')
    print(f"✓ Saved submission to {output_file}")
    
    return df_submission