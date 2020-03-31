import torch
import random


def collate_fn(data):
    if isinstance(data[0], torch.Tensor):
        ispad = False

        for i in data[1:]:
            if i.size() != data[0].size():
                ispad = True
                break

        if ispad:
            data = list(data)
            maxlength = 0
            for i in data:
                maxlength = max(maxlength, len(i))

            for i, tensor in enumerate(data):
                length = tensor.size(0)
                if length == maxlength:
                    continue
                tmp_tensor = tensor.data.new(maxlength).fill_(1)
                tmp_tensor[:length] = tensor
                data[i] = tmp_tensor
        out = None
        return torch.stack(data, 0, out=out)
    elif isinstance(data[0], tuple):
        return [collate_fn(samples) for samples in zip(*data)]
    else:
        return data


def set_random_seed(seed, is_cuda):
    if seed > 0:
        torch.manual_seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    if is_cuda and seed > 0:
        torch.cuda.manual_seed(seed)


def Levenshtein_Distance(str1, str2):
    """
    计算字符串 str1 和 str2 的编辑距离
    :param str1
    :param str2
    :return:
    """
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1

            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1)][len(str2)]
