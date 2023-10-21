import logging
import os
import csv
import sys
sys.path.append('../')

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils import compute_metrics


from load_data import load_and_cache_examples

logger = logging.getLogger(__name__)


def evaluate(args, model, tokenizer, phase):
        
    results = {}

    if phase == "dev":
        eval_dataset = load_and_cache_examples(
            args, 
            tokenizer, 
            data_type = 'dev',
        )

    if phase == "test":
        eval_dataset = load_and_cache_examples(
            args, 
            tokenizer, 
            data_type = 'test',
        )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=args.eval_batch_size,
    )


    # Eval!
    if phase == "dev":
        logger.info("***** Running evaluation*****")
        desc = "Evaluating"
    if phase == "test":
        logger.info("***** Running Inference*****")
        desc = "Infering"

    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in tqdm(eval_dataloader, desc=desc):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
               "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }
            inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                    out_label_ids,
                    inputs['labels'].detach().cpu().numpy(),
                    axis = 0,
                )

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    result = compute_metrics(preds, out_label_ids, phase=phase)
    results.update(result)

    if phase == 'train':
        logger.info("***** Eval results *****")
    if phase == 'test':
        logger.info("***** Infer results *****")

    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        print("  %s = %s" % (key, str(result[key])))


    return results


