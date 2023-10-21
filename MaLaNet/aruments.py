import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--num_hidden_layers",
        default=None,
        type=int,
        required=True,
        help='The number of layers to import.',
    )
    parser.add_argument(
        "--num_output_layers",
        default=None,
        type=int,
        required=True,
        help='The number of layers to output.',
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files "\
             "(or other data files) for the task.",
    )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name.",
    )    
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions"
             "and checkpoints will be written.",
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the logs will be written.",
    )

    # Other parameters  
    parser.add_argument(
        "--load",
        default=None,
        type=str,
        help="The path of ckpts used to continue training."
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization."\
             "Sequences longer than this will be truncated, "\
             "sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", \
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", \
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_infer", action="store_true",\
                        help="Whether to run infer on the test set.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=8,
        type=int,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=1,
        type=int,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing "\
             "a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument("--weight_decay", default=0.01, type=float, \
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, \
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, \
                        help="Max gradient norm.")
    parser.add_argument("--early_stop_steps", type=int, default=500,
                        help="Log every X updates steps.")
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. "\
             "Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, \
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_rate", default=0, type=float, \
                        help="Linear warmup over warmup_rate.")

    parser.add_argument("--logging_steps", type=int, default=500,\
                        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix"\
             "as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument("--seed", type=int, default=6, 
                        help="random seed for initialization")

    args = parser.parse_args()

    return args    
