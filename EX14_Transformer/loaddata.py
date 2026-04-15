import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List
import re

def padding_attention_bias(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    将 (batch, seq_len) 的 1/0 mask 转为可传入 attention(mask=...) 的加性掩码
    """
    bias = torch.zeros_like(attention_mask, dtype=torch.float, device=attention_mask.device)
    bias.masked_fill_(attention_mask == 0, float('-inf'))
    return bias.unsqueeze(1).unsqueeze(2)

class IMDbDataset(Dataset):
    """
    标准的 IMDb 影评情感分析数据集
    使用 aclImdb 数据文件夹
    """
    def __init__(self, data_dir: str, split: str = 'train', max_length: int = 128, vocab=None):
        super().__init__()
        self.max_length = max_length
        self.data = []
        
        split_dir = os.path.join(data_dir, split)
        
        # 加载正面和负面影评
        for label_type, label in [('pos', 1), ('neg', 0)]:
            dir_name = os.path.join(split_dir, label_type)
            if not os.path.exists(dir_name):
                continue
                
            for fname in os.listdir(dir_name):
                if fname.endswith('.txt'):
                    with open(os.path.join(dir_name, fname), 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        self.data.append({
                            'text': text,
                            'label': label
                        })
        
        # 构建或复用词表
        if vocab is None:
            self._build_vocab()
        else:
            self.word2idx = vocab['word2idx']
            self.idx2word = vocab['idx2word']
            self.vocab_size = vocab['vocab_size']
            
        # 编码所有文本
        self._encode_data()
        
        print(f"[OK] {split} 数据集加载完成:")
        print(f"   总样本数: {len(self.data)}")
        print(f"   正面样本: {sum(1 for l in self.labels if l == 1)}")
        print(f"   负面样本: {sum(1 for l in self.labels if l == 0)}")
        print(f"   词汇表大小: {self.vocab_size}")

    def _tokenize(self, text: str) -> List[str]:
        # 转小写并用空格替换非字母数字字符
        text = text.lower()
        # 把 <br /> 替换成空格
        text = text.replace("<br />", " ")
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text.split()

    def _build_vocab(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<CLS>': 2, '<SEP>': 3, '<BOS>': 4, '<EOS>': 5}
        idx = len(self.word2idx)
        
        word_counts = {}
        for row in self.data:
            for word in self._tokenize(row['text']):
                word_counts[word] = word_counts.get(word, 0) + 1
                
        # 频率至少为 5 的词保留（IMDb较大，避免过大的词表）
        for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
            if count >= 5 and word not in self.word2idx:
                self.word2idx[word] = idx
                idx += 1
                
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def encode(self, text: str, max_length: int) -> Dict[str, List[int]]:
        words = self._tokenize(text)
        ids = [self.word2idx['<CLS>']]
        
        for word in words[:max_length-3]:
            ids.append(self.word2idx.get(word, self.word2idx['<UNK>']))
            
        ids.append(self.word2idx['<SEP>'])
        ids.append(self.word2idx['<EOS>'])
        
        padding_length = max_length - len(ids)
        if padding_length > 0:
            ids.extend([self.word2idx['<PAD>']] * padding_length)
            
        attention_mask = [1] * min(len(ids), max_length)
        attention_mask += [0] * max(0, max_length - len(attention_mask))
        
        return {
            'input_ids': ids[:max_length],
            'attention_mask': attention_mask[:max_length]
        }

    def _encode_data(self):
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        
        for row in self.data:
            encoded = self.encode(row['text'], self.max_length)
            self.input_ids.append(encoded['input_ids'])
            self.attention_masks.append(encoded['attention_mask'])
            self.labels.append(row['label'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def create_dataloaders(
    data_dir: str = 'aclImdb',
    batch_size: int = 32,
    max_length: int = 128,
    num_workers: int = 0
) -> tuple:
    # 先加载 train 以构建词表
    train_dataset = IMDbDataset(data_dir, split='train', max_length=max_length)
    
    # 获取建好的词表
    vocab = {
        'word2idx': train_dataset.word2idx,
        'idx2word': train_dataset.idx2word,
        'vocab_size': train_dataset.vocab_size
    }
    
    # 测试集用训练集的词表
    test_dataset = IMDbDataset(data_dir, split='test', max_length=max_length, vocab=vocab)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    print(f"\n[split] 数据划分:")
    print(f"   训练集批次数: {len(train_loader)}")
    print(f"   测试集批次数: {len(test_loader)}")
    
    return train_loader, test_loader, train_dataset
