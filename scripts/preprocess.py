import torch
import random
from sklearn.model_selection import train_test_split
from config import Config
from scripts.utils import load_pairs, seqs_to_tensor

def main():
    random.seed(Config.SEED)
    pairs = load_pairs(Config.RAW_DATA.replace('.csv', '_with_neg.csv'))
    features = []
    labels = []
    for p in pairs:
        tensor = seqs_to_tensor(p['exon1_seq'], p['intron_seq'], p['exon2_seq'])
        features.append(tensor)
        labels.append(int(p['label']))

    features = torch.stack(features)
    labels = torch.tensor(labels, dtype=torch.long)

    # 分层划分
    train_idx, test_idx = train_test_split(
        range(len(labels)), test_size=0.2, stratify=labels, random_state=Config.SEED
    )
    train_val, test = features[train_idx], features[test_idx]
    train_val_labels, test_labels = labels[train_idx], labels[test_idx]

    train, val, train_lbl, val_lbl = train_test_split(
        train_val, train_val_labels, test_size=0.1, stratify=train_val_labels, random_state=Config.SEED
    )

    torch.save((train, train_lbl), Config.TRAIN_FILE)
    torch.save((val, val_lbl), Config.VAL_FILE)
    torch.save((test, test_labels), Config.TEST_FILE)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

if __name__ == '__main__':
    main()
