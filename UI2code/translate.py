import torch
import math
import torch.nn as nn
from torch import Tensor
from utils import get_root_logger, print_log, parse_args, build_model, Levenshtein_Distance, \
    set_random_seed, collate_fn
from data import UIDataset
from torch.utils.data import DataLoader
import os


def translate(model, dataset, max_num_tokens, beam_size, batch_size=1, use_encoder_final=False, out_file=None, cuda=False):
    bos = 2
    eos = 3

    pred_score_total, pred_words_total = 0, 0
    accuracy, num_samples = 0, 0

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2, collate_fn=collate_fn)
    for datas, _, targets_eval, imgPaths in data_loader:
        if cuda:
            datas = datas.cuda()
            targets_eval = targets_eval.cuda()
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        batch_size = datas.size(0)
        num_samples += batch_size
        scores = [[] for _ in range(batch_size)]
        predictions = [[] for _ in range(batch_size)]

        alive_seq = torch.full([batch_size * beam_size, 1], bos, dtype=torch.long, device=device)  # 记录预测序列

        with torch.no_grad():
            if beam_size == 1:  # 贪心
                enc_state, context = model.encoder(datas)
                model.decoder.init_state(enc_state, use_encoder_final=use_encoder_final)

                select_indices = torch.arange(batch_size, dtype=torch.long, device=device)
                original_batch_idx = torch.arange(batch_size, dtype=torch.long, device=device)

                for step in range(max_num_tokens):
                    dec_in = alive_seq[:, -1].view(1, -1)
                    dec_out, attn = model.decoder(dec_in, context)
                    log_probs = model.generator(dec_out.squeeze(0))

                    topk_scores, topk_ids = log_probs.topk(1, dim=-1)
                    is_finished = topk_ids.eq(eos)
                    alive_seq = torch.cat([alive_seq, topk_ids], -1)

                    if alive_seq.shape[1] == max_num_tokens + 1:
                        is_finished.fill_(1)

                    any_finished = is_finished.any()
                    if any_finished:
                        finished = is_finished.view(-1).nonzero()
                        for b in finished:
                            b_orig = original_batch_idx[b]
                            scores[b_orig].append(topk_scores[b, 0].item())
                            predictions[b_orig].append(alive_seq[b, 1:].squeeze(0))

                        if is_finished.all():
                            break

                        # 更新状态
                        is_alive = ~is_finished.view(-1)
                        alive_seq = alive_seq[is_alive]
                        select_indices = is_alive.nonzero().view(-1)
                        original_batch_idx = original_batch_idx[is_alive]

                        context = context.index_select(1, select_indices)
                        model.decoder.map_state(
                            lambda state, dim: state.index_select(dim, select_indices))
            else:
                enc_state, context = model.encoder(datas)
                model.decoder.init_state(enc_state, use_encoder_final=use_encoder_final)
                model.decoder.map_state(
                    lambda state, dim: tile(state, beam_size, dim=dim))
                context = tile(context, beam_size, dim=1)

                top_beam_finished = torch.zeros([batch_size], dtype=torch.uint8, device=device).bool()
                beam_offset = torch.arange(0, batch_size * beam_size, step=beam_size, dtype=torch.long, device=device)
                batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)

                hypotheses = [[] for _ in range(batch_size)]
                topk_scores = torch.tensor([0.0] + [float('-inf')] * (beam_size - 1), device=device).repeat(batch_size)
                topk_ids = torch.empty((batch_size, beam_size), dtype=torch.long, device=device)
                batch_index = torch.empty((batch_size, beam_size), dtype=torch.long, device=device)

                for step in range(max_num_tokens):
                    dec_in = alive_seq[:, -1].view(1, -1)
                    dec_out, attn = model.decoder(dec_in, context)
                    log_probs = model.generator(dec_out.squeeze(0))

                    vocab_size = log_probs.size(-1)
                    B = log_probs.size(0) // beam_size

                    log_probs += topk_scores.view(B * beam_size, 1)
                    log_probs = log_probs.reshape(B, beam_size * vocab_size)
                    torch.topk(log_probs, beam_size, dim=-1, out=(topk_scores, topk_ids))  # 获取当前累计最高scores, ids

                    torch.div(topk_ids, vocab_size, out=batch_index)
                    batch_index += beam_offset[:B].unsqueeze(1)
                    select_indices = batch_index.view(B * beam_size)
                    topk_ids.fmod_(vocab_size)  # 获取真实id

                    alive_seq = torch.cat([alive_seq.index_select(0, select_indices),
                                           topk_ids.view(B * beam_size, 1)], -1)
                    is_finished = topk_ids.eq(eos)

                    if alive_seq.shape[1] == max_num_tokens + 1:
                        is_finished.fill_(1)

                    any_finished = is_finished.any()
                    if any_finished:
                        top_beam_finished |= is_finished[:, 0].eq(1)

                        preds = alive_seq.view(B, beam_size, -1)

                        non_finished_batch = []
                        for i in range(is_finished.size(0)):
                            b = batch_offset[i].item()
                            finished = is_finished[i].nonzero().view(-1)
                            for j in finished:
                                hypotheses[b].append((
                                    topk_scores[i, j].item(),
                                    preds[i, j, 1:],))

                            finish_flag = top_beam_finished[i]  # top_beam的分数通常是最高的
                            if finish_flag:
                                best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                                for n, (score, pred) in enumerate(best_hyp):
                                    scores[b].append(score)
                                    predictions[b].append(pred)
                            else:
                                non_finished_batch.append(i)
                        non_finished = torch.tensor(non_finished_batch, device=device)
                        topk_scores.masked_fill_(is_finished, -1e10)

                        if len(non_finished) == 0:
                            break

                        # 更新状态
                        B_new = non_finished.size(0)
                        top_beam_finished = top_beam_finished.index_select(0, non_finished)
                        batch_offset = batch_offset.index_select(0, non_finished)
                        topk_scores = topk_scores.index_select(0, non_finished)
                        batch_index = batch_index.index_select(0, non_finished)
                        select_indices = batch_index.view(B_new * beam_size)
                        alive_seq = preds.index_select(0, non_finished).view(-1, alive_seq.size(-1))
                        topk_ids = topk_ids.index_select(0, non_finished)

                        context = context.index_select(1, select_indices)
                        model.decoder.map_state(
                            lambda state, dim: state.index_select(dim, select_indices))

            translations = []

            for b in range(batch_size):
                tokens = []
                pred_str = []
                for tok in predictions[b][0]:
                    if tok in [1, 2, 3]:
                        break
                    elif tok == 0:
                        l = 'unk'
                    else:
                        l = dataset.id2vocab[tok - 4]
                    tokens.append(l)
                    pred_str += list(l)

                target_label_str = []
                target_label_strlist = []
                target_label_list = targets_eval[b].masked_select(targets_eval[b].ne(1))
                for label in target_label_list:
                    if label == 3:
                        break
                    elif label == 0:
                        l = 'unk'
                    else:
                        l = dataset.id2vocab[label - 4]
                    target_label_strlist.append(l)
                    target_label_str += list(l)
                edit_distance = Levenshtein_Distance(pred_str, target_label_str)
                accuracy += 1 - min(1, edit_distance / float(len(target_label_str)))

                translation = {'pred_sent': tokens, 'pred_score': scores[b][0],
                               'filename': imgPaths[b], 'tgt': target_label_strlist}
                translations.append(translation)

            for trans in translations:
                pred_score_total += trans['pred_score']
                pred_words_total += len(trans['pred_sent'])
                pred = ''
                for p in trans['pred_sent']:
                    pred += p + ' '
                tgt = ''
                for t in trans['tgt']:
                    tgt += t + ' '
                if out_file:
                    out_file.write(trans['filename'])
                    out_file.write('\t')
                    out_file.write(tgt)
                    out_file.write('\t')
                    out_file.write(pred)
                    out_file.write('\t')
                    out_file.write('%.4f' % trans['pred_score'])
                    out_file.write('\n')
                    out_file.flush()

            avg_score = pred_score_total / pred_words_total
            acc = 100 * (accuracy / num_samples)
            ppl = math.exp(-pred_score_total / pred_words_total)
            msg = ("AVG SCORE: %.4f, ACC: %.4f, PPL: %.4f" % (avg_score, acc, ppl))
            print_log(msg)
    if out_file:
        out_file.close()
    avg_score = pred_score_total / pred_words_total
    acc = 100 * (accuracy / num_samples)
    ppl = math.exp(-pred_score_total / pred_words_total)
    msg = ("AVG SCORE: %.4f, ACC: %4.f, PPL: %.4f" % (avg_score, acc, ppl))
    print_log(msg)


def tile(x, count, dim=0):
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1).repeat(1, count).contiguous().view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def main():
    args = parse_args()
    logging = get_root_logger(args.translate_log, mode='w')
    out_file = None
    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        out_file = open(os.path.join(args.output_dir, 'result.txt'), 'w', encoding='utf-8')

    if args.seed:
        set_random_seed(args.seed, args.gpu_id > -1)
    gpu_id = args.gpu_id
    model_path = args.model_path
    beam_size = args.beam_size
    batch_size = args.batch_size
    max_num_tokens = args.max_num_tokens

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
    logging.info('Load testing data from ' + args.data_path)
    dataset = UIDataset(args.data_base_dir, args.data_path, args.label_path, args.vocab_file)

    # Build model
    assert os.path.exists(model_path), 'make sure the model path'
    logging.info('Loading model from %s' % model_path)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

    model = build_model(args, gpu_id, checkpoint)
    model.eval()
    translate(model, dataset, max_num_tokens, beam_size, batch_size, False, out_file, args.cuda)


if __name__ == '__main__':
    main()
