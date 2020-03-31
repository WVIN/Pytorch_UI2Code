import torch
import torch.nn as nn
from model import ImageEncoder
from model import RNNDecoder
from model import UIModel


def build_model(config, gpu_id, checkpoint=None):
    # Build encoder
    encoder = ImageEncoder(config.encoder_num_layers, True, config.encoder_num_hidden,
                           config.dropout, config.image_channel_size)
    # Build decoder
    decoder_num_hidden = config.encoder_num_hidden
    decoder = RNNDecoder(True, config.target_embedding_size, config.decoder_num_layers,
                         decoder_num_hidden, config.dropout, config.target_vocab_size,
                         attn_type='general', input_feed=config.input_feed)

    device = torch.device('cuda') if gpu_id >= 0 else torch.device('cpu')

    # Build Generator
    generator = nn.Sequential(
        nn.Linear(decoder_num_hidden, config.target_vocab_size),
        nn.LogSoftmax(dim=-1)
    )

    # Build UIModel
    model = UIModel(encoder, decoder, generator)

    # Load the model states from checkpoint or initialize them
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
    else:
        for p in model.parameters():
            p.data.uniform_(-config.param_init, config.param_init)

    model.to(device)
    return model
