
import json
import logging
import os
import random
import sys
sys.path.append('../')

from arguments import get_args

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import fitlog

import transformers
from transformers import WEIGHTS_NAME
from transformers import BertTokenizer as ElasticBertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import is_main_process

from models.configuration_elasticbert import ElasticBertConfig
from models.modeling_elasticbert_entropy import ElasticBertForSequenceClassification, S_Net

from evaluations import evaluate
from load_data import load_and_cache_examples

# from elue import elue_output_modes, elue_processors
from utils import TStega_processor


logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def train(args, train_dataset, model, tokenizer):
    fitlog.set_log_dir(args.log_dir)
    fitlog.add_hyper(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir) 

    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, \
        batch_size=args.train_batch_size)    

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) //\
            args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps \
            * args.num_train_epochs


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not \
                       any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if \
                    any(nd in n for nd in no_decay)], 
         "weight_decay": 0.0
         },
    ]   

    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr = args.learning_rate, 
        eps = args.adam_epsilon,
    )
    
    if args.warmup_steps > 0:
        num_warmup_steps = args.warmup_steps
    else:
        assert args.warmup_rate != 0.0
        num_warmup_steps = args.warmup_rate * t_total

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=t_total,
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))\
        and os.path.isfile(os.path.join(args.model_name_or_path, 
                           "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(
            args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(
            args.model_name_or_path, "scheduler.pt")))


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", 
        args.train_batch_size
    )
    logger.info(
        "Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps,
    )
    logger.info("  Gradient Accumulation steps = %d", 
        args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)


    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if args.load is not None:
        if os.path.exists(args.load):
            # set global_step to gobal_step of last saved checkpoint 
            # from model path
            global_step = int(args.load.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (len(train_dataloader) // \
                             args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % \
                    (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("Continuing training from checkpoint,"\
                        "will skip to saved global_step")
            logger.info("Continuing training from epoch %d", epochs_trained)
            logger.info("Continuing training from global step %d", global_step)
            logger.info(
                "  Will skip the first %d steps in the first epoch",
                steps_trained_in_current_epoch,
            ) 

    best_all_metric = {}
    keep_best_step = 0
    tr_loss, logging_loss, best = 0.0, 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=False,
    )
    set_seed(args)  # Added here for reproductibility
    metric_key = 'accuracy'


    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, 
            desc="Iteration", 
            disable=False,
        )
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }
            inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            loss = outputs[0] 

            # output = model(batch[0])
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()    

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                args.max_grad_norm,
                )

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 \
                    and global_step % args.logging_steps == 0:
                    logs = {}
                    if args.evaluate_during_training: 
                        results = evaluate(args, model, tokenizer, 'dev')
                        res_for_display = {}
                        num_metric = 0
                        avg_metric = 0
                        for k, v in results.items():
                            num_metric += 1
                            avg_metric += v
                            res_for_display[k.replace('-', '_')] = v
                            
                        fitlog.add_metric(
                            {"dev": res_for_display},
                              step = global_step,
                        )

                        if results[metric_key] > best:
                            keep_best_step = 0
                            best = results[metric_key]
                            best_all_metric.update(results)
                            fitlog.add_best_metric(
                                {'dev': {metric_key.replace('-','_'): best}}
                            )
                           
                           # Save the best model
                            output_dir = os.path.join(
                                args.output_dir, 'best_model'
                            )
                            model_to_save = (
                                model.module if hasattr(model,"module") \
                                else model)

                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)
                            torch.save(
                                args, 
                                os.path.join(output_dir, 'training_args.bin'),
                            )
                            logger.info(
                                "Saving model checkpoint to %s", output_dir,
                            )

                            torch.save(optimizer.state_dict(),
                                os.path.join(output_dir, "optimizer.pt")
                            )
                            torch.save(scheduler.state_dict(),
                                os.path.join(output_dir, "scheduler.pt")
                            )
                            logger.info(
                            "Saving optimizer and scheduler state to %s",
                            output_dir,
                            )
                        
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    fitlog.add_loss(
                        loss_scalar,
                        name = "Loss",
                        step = global_step,
                    )

                    print(json.dumps({**logs, **{"step": global_step}}))

                    if keep_best_step >= args.early_stop_steps:
                        epoch_iterator.close()
                        break

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if (args.evaluate_during_training and args.logging_steps == 0):
            keep_best_step += 1
            logs = {}
            results = evaluate(args, model, tokenizer, 'dev')
            res_for_display = {}
            for k, v in results.items():
                res_for_display[k.replace('-','_')] = v
            fitlog.add_metric({'dev':res_for_display}, step=global_step)
            if results[metric_key] > best:
                keep_best_step = 0
                best = results[metric_key]
                best_all_metric.update(results)
                fitlog.add_best_metric({'dev':{metric_key.replace('-','_'):best}})

                # Save the best model
                output_dir = os.path.join(args.output_dir, 'best_model')
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                torch.save(
                    args, os.path.join(output_dir, "training_args.bin")
                )
                logger.info("Saving model checkpoint to %s", output_dir)

                torch.save(optimizer.state_dict(),
                    os.path.join(output_dir, "optimizer.pt",
                    )
                )
                torch.save(scheduler.state_dict(),
                    os.path.join(output_dir, "scheduler.pt"
                    )
                )
                logger.info(
                    "Saving optimizer and scheduler states to %s",
                    output_dir,
                )

            for key, value in results.items():
                eval_key = "eval_{}".format(key)
                logs[eval_key] = value

            learning_rate_scalar = scheduler.get_lr()[0]
            logs["learning_rate"] = learning_rate_scalar

            print(json.dumps({**logs, **{"step", global_step}}))

        if keep_best_step >= args.early_stop_steps:
            train_iterator.close()
            logging.info(
                "The task stops early at step {}/".format(global_step)
            )

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    logs = {}
    if (args.evaluate_during_training and args.logging_steps > 0 \
        and global_step % args.logging_steps != 0 and \
        keep_best_step < args.early_stop_steps
    ):
        results = evaluate(args, model, tokenizer, 'dev')
        res_for_display = {}
        for k, v in results.items():
            res_for_display[k.replace('-','_')] = v
        fitlog.add_metric({'dev': res_for_display}, step = global_step)
        if results[metric_key] > best:
            best = results[metric_key]
            best_all_metric.update(results)
            fitlog.add_best_metric({"dev":{metric_key.replace('-','_'):best}})
            # Save the best model
            output_dir = os.path.join(args.output_dir, "beat_model")
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)

            torch.save(
                optimizer.state_dict(),
                os.path.join(output_dir, "optimizer.pt",
                )
            )
            torch.save(
                scheduler.state_dict(),
                os.path.join(output_dir, "scheduler.pt",
                )
            )
            logger.info(
                "Saving optimizer and scheduler state to %s",
                output_dir,
            )

        for key, value in results.items():
            eval_key = "eval_{}".format(key)
            logs[eval_key] = value

        learning_rate_scalar = scheduler.get_lr()[0]
        logs["learning_rate"] = learning_rate_scalar

        print(json.dumps({**logs, **{"step": global_step}}))

    fitlog.finish()

    return global_step, tr_loss / global_step, best


def main():
    args = get_args()


    if not os.path.exists(args.log_dir):
        try:
            os.makedirs(args.log_dir)
        except:
            pass

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. "\
            "Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(-1):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    # Set seed
    set_seed(args)

    # Prepare TSteganalysis data_iter
    processor = TStega_processor()
    label_list = processor.get_labels()
    num_labels = len(label_list)


    # Load pretrained model and tokenizer
    config = ElasticBertConfig.from_pretrained(        
        args.model_name_or_path,
        num_labels = num_labels,
        num_hidden_layers = args.num_hidden_layers,
        num_output_layers = args.num_output_layers,
        cache_dir = None,
    )

    tokenizer = ElasticBertTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case = True,
        cache_dir = None,        
    )

    args.embed_num = tokenizer.vocab_size
    model = ElasticBertForSequenceClassification.from_pretrained(
        args.model_name_or_path, 
        config = config,
        args = args,
        add_pooling_layer = True, 
    )

    # args.embed_num = tokenizer.vocab_size
    # model = S_Net(args)

    model.to(args.device)  

    print("Total Model Parameters:", sum(param.numel() \
          for param in model.parameters()))
    print("################################")
    for name, weight in model.named_parameters():
        if 'classifiers' in name:
            print(name)
    print("################################")


    logger.info("Training/evaluation parameters %s", args)

    train_dataset = None
    best_all_metric = None
    if args.do_train:
        train_dataset = load_and_cache_examples(
            args, 
            tokenizer, 
            data_type='train',
        )
        global_step, tr_loss, best_all_metric  = train(
            args, 
            train_dataset, 
            model, 
            tokenizer
        )
        logger.info(
            " global_step = %s, average loss = %s", 
            global_step, 
            tr_loss,
        )

    if args.do_infer:
        best_model_path = os.path.join(args.output_dir, "best_model")
        if os.path.exists(best_model_path):
            model = ElasticBertForSequenceClassification.from_pretrained(
                best_model_path,
                args = args,
            )
            model.to(args.device)
            results = evaluate(args, model, tokenizer, 'test')

            test_result_file = os.path.join("./test_results.txt")
            with open(test_result_file, 'a', encoding='utf-8') as fout:
                fout.write(args.output_dir + '\n')
                for key in results.keys():
                    fout.write(key + '\t' + str(results[key]) + '\n')
        else:
            raise Exception("There is not best model path.")

    return best_all_metric
            
    


if __name__ == "__main__":
    best = main()

