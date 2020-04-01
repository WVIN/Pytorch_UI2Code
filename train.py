import torch
import math
import torch.nn as nn
from utils import get_root_logger, print_log, parse_args, build_model, Optimizer, ModelSaver, \
    collate_fn, set_random_seed
from data import UIDataset
from torch.utils.data import DataLoader


def train(model, optim, model_saver, num_epochs, train_loader, val_loader, steps_per_checkpoint,
          valid_steps, lr_decay, start_decay_at, cuda):
    loss = 0
    num_corrects = 0
    num_nonzeros = 0
    learning_rate = optim.optimizer.param_groups[0]['lr']
    print_log('Lr: %f' % learning_rate)
    criterion = nn.NLLLoss(ignore_index=1, reduction='sum')
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        if epoch >= start_decay_at and learning_rate > args.learning_rate_min:
            learning_rate = learning_rate * lr_decay
            optim.optimizer.param_groups[0]['lr'] = max(learning_rate, args.learning_rate_min)
            print_log('Decay lr, current lr: %f' % learning_rate)
        for batch_datas, batch_targets, batch_targets_eval, _ in train_loader:
            if cuda:
                batch_datas = batch_datas.cuda()
                batch_targets = batch_targets.cuda()
                batch_targets_eval = batch_targets_eval.cuda()

            optim.zero_grad()
            scores = model(batch_datas, batch_targets, use_encoder_final=False)

            target = batch_targets_eval.transpose(0, 1).contiguous().view(-1)

            step_loss = criterion(scores, target) / float(batch_datas.size(0))

            pred = scores.max(1)[1]
            non_padding = target.ne(1)
            num_correct = pred.eq(target).masked_select(non_padding).sum().item()
            num_non_padding = non_padding.sum().item()
            num_corrects += num_correct
            num_nonzeros += num_non_padding

            loss = loss + step_loss.item()
            if optim.training_step % 50 == 0:
                print_log('Epoch: %d, Step: %d, Loss: %f, lr: %f, acc: %f, ppl: %f' %
                          (epoch, optim.training_step, step_loss, optim.optimizer.param_groups[0]['lr'],
                           100 * (num_corrects / num_nonzeros), math.exp(min(loss / num_nonzeros, 100))))

                num_corrects = 0
                num_nonzeros = 0
                loss = 0

            if optim.training_step % steps_per_checkpoint == 0:
                ckpth = model_saver.save(optim.training_step)
                print_log('Saving model at %s' % ckpth)

            optim.backward(step_loss)
            optim.step()

            if model.decoder.state is not None:
                model.decoder.detach_state()

            if val_loader and optim.training_step % valid_steps == 0:
                # Evaluate on val data
                print_log('Evaluating model')
                val_loss = 0
                val_num_corrects = 0
                val_num_nonzeros = 0

                model.eval()
                with torch.no_grad():
                    for val_batch_datas, val_batch_targets, val_batch_targets_eval, _ in val_loader:
                        if cuda:
                            val_batch_datas = val_batch_datas.cuda()
                            val_batch_targets = val_batch_targets.cuda()
                            val_batch_targets_eval = val_batch_targets_eval.cuda()

                        val_scores = model(val_batch_datas, val_batch_targets, use_encoder_final=False)
                        val_target = val_batch_targets_eval.transpose(0, 1).contiguous().view(-1)
                        val_step_loss = criterion(val_scores, val_target) / float(batch_datas.size(0))

                        val_pred = val_scores.max(1)[1]
                        val_non_padding = val_target.ne(1)
                        val_num_correct = val_pred.eq(val_target).masked_select(val_non_padding).sum().item()
                        val_num_non_padding = val_non_padding.sum().item()

                        val_num_corrects += val_num_correct
                        val_num_nonzeros += val_num_non_padding

                        val_loss += val_step_loss.item()
                val_losses.append(val_loss)
                print_log('Epoch: %d Step %d - Val Ppl = %f - Accuracy = %f'
                          % (epoch, optim.training_step, math.exp(min(val_loss / val_num_nonzeros, 100)),
                             100 * (val_num_corrects / val_num_nonzeros)))

                if len(val_losses) > 1 and val_losses[-1] > val_losses[-2] and learning_rate > args.learning_rate_min:
                    learning_rate = learning_rate * lr_decay
                    optim.optimizer.param_groups[0]['lr'] = max(learning_rate, args.learning_rate_min)
                    print_log('Decay lr, current lr: %f' % learning_rate)

                model.train()


def configure_process(cfg, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(cfg.seed, device_id >= 0)


def cal_parameters(model):
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        if 'decoder' in name:
            dec += param.nelement()
    return enc + dec, enc, dec


args = parse_args()


def main():
    logging = get_root_logger(args.log_path, mode='a')
    logging.info('Command Line Arguments:')
    for key, i in vars(args).items():
        logging.info(key + ' = ' + str(i))
    logging.info('End Command Line Arguments')

    batch_size = args.batch_size
    num_epochs = args.num_epochs

    resume_from = args.resume_from
    steps_per_checkpoint = args.steps_per_checkpoint

    gpu_id = args.gpu_id

    configure_process(args, gpu_id)
    if gpu_id > -1:
        logging.info('Using CUDA on GPU ' + str(gpu_id))
        args.cuda = True
    else:
        logging.info('Using CPU')
        args.cuda = False

    '''Load data'''
    logging.info('Data base dir ' + args.data_base_dir)
    logging.info('Loading vocab from ' + args.vocab_file)
    with open(args.vocab_file, "r", encoding='utf-8') as f:
        args.target_vocab_size = len(f.readlines()) + 4
    logging.info('Load training data from ' + args.data_path)
    train_data = UIDataset(args.data_base_dir, args.data_path, args.label_path, args.vocab_file)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=2, drop_last=True, collate_fn=collate_fn)

    logging.info('Load validation data from ' + args.val_data_path)
    val_data = UIDataset(args.data_base_dir, args.val_data_path, args.label_path, args.vocab_file)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True,
                            num_workers=2, drop_last=True, collate_fn=collate_fn)

    # Build model
    logging.info('Building model')
    if args.resume_from:
        logging.info('Loading checkpoint from %s' % resume_from)
        checkpoint = torch.load(resume_from)
    else:
        checkpoint = None
        logging.info('Creating model with fresh parameters')
    model = build_model(args, gpu_id, checkpoint)
    logging.info(model)

    n_params, enc, dec = cal_parameters(model)
    logging.info('encoder: %d' % enc)
    logging.info('decoder: %d' % dec)
    logging.info('number of parameters: %d' % n_params)

    # Build optimizer
    optimier = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    optim = Optimizer(optimier)
    if checkpoint:
        optim.load_state_dict(checkpoint['optim'])
        optim.training_step += 1

    # Build model saver
    model_saver = ModelSaver(args.model_dir, model, optim)

    train(model, optim, model_saver, num_epochs, train_loader, val_loader, steps_per_checkpoint,
          args.valid_steps, args.lr_decay, args.start_decay_at, args.cuda)


if __name__ == '__main__':
    main()
