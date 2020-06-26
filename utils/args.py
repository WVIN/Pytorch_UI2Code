import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='UI2code')
    '''Input and Output'''
    parser.add_argument('--data_base_dir', type=str, default='',
                        help='The base directory of the image path in data-path. If the image path in data-path is '
                             'absolute path, set it to /')
    parser.add_argument('--data_path', type=str, default='',
                        help='The path containing data file names and labels. Format per line: image_path characters')
    parser.add_argument('--label_path', type=str, default='',
                        help='The path containing data file names and labels. Format per line: image_path characters')
    parser.add_argument('--val_data_path', type=str, default='',
                        help='The path containing validate data file names and labels. Format per line: image_path '
                             'characters')
    parser.add_argument('--model_dir', type=str, default='checkpoints',
                        help='The directory for saving and loading model parameters (structure is not stored)')
    parser.add_argument('--log_path', type=str, default='log.txt', help='The path to put log')
    parser.add_argument('--translate_log', type=str, default='translate_log.txt', help='The path to put translate log')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='The path to put visualization results if visualize is set to True')

    '''Display'''
    parser.add_argument('--steps_per_checkpoint', type=int, default=2000,
                        help='Checkpointing (print perplexity, save model) per how many steps')
    parser.add_argument('--beam_size', type=int, default=2, help='Beam size.')

    '''argsimization'''
    parser.add_argument('--valid_steps', type=int, default=6000, help='Performing validation every X steps')
    parser.add_argument('--num_epochs', type=int, default=10, help='The number of whole data passes')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.00001, help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5,
                        help='Decay learning rate by this much if (i) perplexity does not decrease on the validation '
                             'set or (ii) epoch has gone past the start_decay_at_limit')
    parser.add_argument('--start_decay_at', type=int, default=999, help='Start decay after this epoch')

    '''Network'''
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--target_embedding_size', type=int, default=80, help='Embedding dimension for each target')
    parser.add_argument('--input_feed', type=bool, default=True, help='Whether or not use LSTM attention decoder cell')
    parser.add_argument('--encoder_num_hidden', type=int, default=512, help='Number of hidden units in encoder cell')
    parser.add_argument('--encoder_num_layers', type=int, default=1,
                        help='Number of hidden layers in encoder cell')  # does not support >1 now!!!
    parser.add_argument('--decoder_num_layers', type=int, default=1, help='Number of hidden units in decoder cell')
    parser.add_argument('--vocab_file', type=str, default='dataset/xml_vocab.txt', help='Vocabulary file. A token per line.')
    parser.add_argument('--image_channel_size', type=int, default=1, choices=[3, 1],
                        help="Using grayscale image can training "
                        "model faster and smaller")

    '''Other'''
    parser.add_argument('--model_path', type=str, default='',
                        help="Path to model .pt file(s). ")
    parser.add_argument('--gpu_id', type=int, default=0, help='Which gpu to use. <0 means use CPU')
    parser.add_argument('--resume_from', default='', type=str,
                        help='If resume training from a checkpoint then this is the '
                             'path to the pretrained model\'s state_dict.')
    parser.add_argument('--param_init', '-param_init', type=float, default=0.05,
                        help="Parameters are initialized over uniform distribution "
                             "with support (-param_init, param_init). "
                             "Use 0 to not use initialization")
    parser.add_argument('--seed', type=int, default=910820, help='Random seed')
    parser.add_argument('--max_num_tokens', type=int, default=100,
                        help='Maximum number of output tokens')  # when evaluate, this is the cut-off length.
    return parser.parse_args()
