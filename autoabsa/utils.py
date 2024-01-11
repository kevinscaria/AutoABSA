import torch
import pandas as pd


def get_device():
    return torch.device(
        'mps' if torch.backends.mps.is_built() else
        'cuda' if torch.cuda.is_available() else
        'cpu'
    )


def get_df(data_path, sample_size=1.0):
    df = pd.read_json(data_path)
    df['aspect_term'] = df['aspects'].apply(lambda x: [' '.join(i['term']) for i in x])
    df['opinion_word'] = df['opinions'].apply(lambda x: [' '.join(i['term']) for i in x])
    df['polarity'] = df['aspects'].apply(lambda x: [''.join(i['polarity']) for i in x])
    df = df.explode(column=['aspect_term', 'opinion_word', 'polarity']).reset_index(drop=True)
    df = df[['raw_words', 'aspect_term', 'opinion_word', 'polarity']]
    df = df.sample(frac=sample_size, random_state=42)
    return df


def get_metrics(y_true, y_pred, is_triplet_extraction=False, is_partial=True):
    total_pred = 0
    total_gt = 0
    tp = 0
    if not is_triplet_extraction:
        for gt, pred in zip(y_true, y_pred):
            gt_list = gt.split(', ')
            pred_list = pred.split(', ')
            total_pred += len(pred_list)
            total_gt += len(gt_list)
            for gt_val in gt_list:
                for pred_val in pred_list:
                    if is_partial:
                        if pred_val in gt_val or gt_val in pred_val:
                            tp += 1
                            break
                    else:
                        if pred_val == gt_val or gt_val == pred_val:
                            tp += 1
                            break

    else:
        for gt, pred in zip(y_true, y_pred):
            gt_list = gt.split(', ')
            pred_list = pred.split(', ')
            total_pred += len(pred_list)
            total_gt += len(gt_list)
            for gt_val in gt_list:
                gt_asp = gt_val.split(':')[0]

                try:
                    gt_op = gt_val.split(':')[1]
                except:
                    continue

                try:
                    gt_sent = gt_val.split(':')[2]
                except:
                    continue

                for pred_val in pred_list:
                    pr_asp = pred_val.split(':')[0]

                    try:
                        pr_op = pred_val.split(':')[1]
                    except:
                        continue

                    try:
                        pr_sent = gt_val.split(':')[2]
                    except:
                        continue

                    if is_partial:
                        if pr_asp in gt_asp and pr_op in gt_op and gt_sent == pr_sent:
                            tp += 1
                    else:
                        if pr_asp == gt_asp and pr_op == gt_op and gt_sent == pr_sent:
                            tp += 1

    p = tp / total_pred
    r = tp / total_gt
    return p, r, 2 * p * r / (p + r), None
