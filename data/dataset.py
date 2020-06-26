import os
import torch
from torch.utils import data
import cv2
import numpy as np
from torchvision import transforms as T
from tqdm import tqdm

transforms = T.Compose([
    T.ToTensor(),
    # T.Normalize(mean=[.5], std=[.5])
    # T.Normalize(mean=[0.64534397, 0.65154537, 0.6550537], std=[0.21804649, 0.20329148, 0.20559017])
])


class UIDataset(data.Dataset):
    def __init__(self, data_base_dir, data_path, label_path, vocab_path):
        self.data_base_dir = data_base_dir
        self.label_path = label_path
        self.imgs = []
        self.labels = []
        self.id2vocab = []
        self.vocab2id = {}
        self.transforms = transforms
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                self.id2vocab.append(line.split('\n')[0])

        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                self.labels.append(line.split('\n')[0])

        with open(data_path, 'r') as f:
            for line in f.readlines():
                imginfo = {'filename': line.split()[0], 'label_id': int(line.split()[1])}
                self.imgs.append(imginfo)

        for i in range(len(self.id2vocab)):
            self.vocab2id[self.id2vocab[i]] = i + 4

    '''返回一个样本数据'''
    def __getitem__(self, item):
        img_path = os.path.join(self.data_base_dir, self.imgs[item]['filename'])  # 第item张图片的路径
        numlist = [2]  # 加上开始id
        strlist = self.labels[self.imgs[item]['label_id']].split()  # 获取标签信息
        for i in range(len(strlist)):  # 将标签信息转换为对应的id列表,0:空值, 1:默认填充值, 2:开始, 3:结尾
            token = strlist[i]
            if self.vocab2id[token]:
                numlist.append(self.vocab2id[token])
            else:
                numlist.append(0)
        target = torch.tensor(numlist.copy())
        numlist.append(3)
        img = cv2.imread(img_path, 0)  # 读入图片
        if self.transforms:
            img = self.transforms(img)
        target_eval = torch.tensor(numlist)[1:]
        return img, target, target_eval, img_path.split('\\')[-1]  # 返回图片对应的tensor及其标签

    '''样本的数量'''
    def __len__(self):
        return len(self.imgs)


def mean_std(root_path):
    imgnames = os.listdir(root_path)
    m_list, s_list = [], []
    for img_filename in tqdm(imgnames):
        img = cv2.imread(os.path.join(root_path, img_filename))
        # img = img / 255.0
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print(m[0][::-1])
    print(s[0][::-1])


if __name__ == '__main__':
    '''获取样本均值和标准差'''
    # mean_std(data_base_dir)
