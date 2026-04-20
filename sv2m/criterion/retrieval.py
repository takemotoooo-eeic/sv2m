import numpy as np


def retrieval_metrics(sim_matrix: np.ndarray, all_music_ids_list: list[str]) -> tuple[dict, np.ndarray, dict]:
    '''
    Input:
        sim_matrix: [val_len, val_len] - The raw similarity matrix
        all_music_ids_list: [val_len] - The list of actual music IDs corresponding to each video
    Return:
        metrics: dict
        ind: np.array [val_len] - The position of each video's music ID in the sorted matrix
        topk_music_ids: dict - The topk music IDs corresponding to each video
    '''
    # When multiple videos correspond to the same music ID, this part will remove duplicate music IDs 
    # to more accurately assess the ranking of the GT (ground truth) music ID.
    # Get the indices of the sorted similarity matrix in descending order
    sort_indices = np.argsort(sim_matrix, axis=1)[:, ::-1]  # [val_len, val_len]
    
    ret_results_list = []
    ind = []
    for i, gt_music_id in enumerate(all_music_ids_list):
        seen_music_ids = set()  # Set used to track already encountered music IDs
        sorted_music_ids = [all_music_ids_list[idx] for idx in sort_indices[i]]
        # Find the position of the GT music ID in the sorted list
        for music_id in sorted_music_ids:
            if music_id not in seen_music_ids:
                seen_music_ids.add(music_id)
                if music_id == gt_music_id:
                    now_ind = len(seen_music_ids) - 1
                    ind.append(now_ind)  # Add the current music ID's ranking position after deduplication
                    break
        pred_dict_np = dict(
            music_id = gt_music_id,
            rank = now_ind + 1,
            topk_music_ids = sorted_music_ids[:1]
        )
        ret_results_list.append(pred_dict_np)
    ind = np.array(ind)  # [val_len]
    assert len(ind) == len(all_music_ids_list), "len(ind) != len(all_music_ids_list)"
    
    # Calculate the evaluation metrics
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R3'] = float(np.sum(ind < 3)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['R20'] = float(np.sum(ind < 20)) * 100 / len(ind)
    metrics['R25'] = float(np.sum(ind < 25)) * 100 / len(ind)
    metrics['R50'] = float(np.sum(ind < 50)) * 100 / len(ind)
    metrics['R100'] = float(np.sum(ind < 100)) * 100 / len(ind)
    metrics["MedianR"] = np.median(ind) + 1
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    # Compute MRR (Mean Reciprocal Rank)
    reciprocal_ranks = 1.0 / (ind + 1)
    metrics['MRR'] = np.mean(reciprocal_ranks)

    return metrics, ind, ret_results_list