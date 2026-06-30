import csv
import sys   # 如果没有导入sys，请添加
import numpy as np
import torch
from config import Config

csv.field_size_limit(sys.maxsize)   # <-- 添加这一行

def dna_to_onehot(seq, max_len=Config.MAX_SEQ_LEN):
    """将DNA序列转为one-hot矩阵 (4, max_len)"""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq = seq.upper()[:max_len]
    mat = np.zeros((4, max_len), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in mapping:
            mat[mapping[base], i] = 1.0
    return mat

def seqs_to_tensor(exon1, intron, exon2, max_len=Config.MAX_SEQ_LEN):
    """将三个序列拼接成 (3,4,max_len) 的张量（三通道）"""
    e1 = dna_to_onehot(exon1, max_len)
    i1 = dna_to_onehot(intron, max_len)
    e2 = dna_to_onehot(exon2, max_len)
    return torch.tensor(np.stack([e1, i1, e2]), dtype=torch.float32)

def load_pairs(path):
    """从TSV读取正样本，返回列表[{exon1_seq, intron_seq, exon2_seq, type}]"""
    import csv
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for r in reader:
            rows.append(r)
    return rows