import json
import torch
from typing import Dict, List, Union

class GeneTokenizer:
    def __init__(self, vocab_file: str, max_length: int = 512):
        self.max_length = max_length
        self.vocab = self._load_vocab(vocab_file)
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # 设定特殊 token 和 ID
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"

        self.pad_token_id = self.vocab.get(self.pad_token, 0)
        self.mask_token_id = self.vocab.get(self.mask_token, 1)
        self.unk_token_id = self.vocab.get(self.unk_token, 2)
        self.cls_token_id = self.vocab.get(self.cls_token, 3)
        self.sep_token_id = self.vocab.get(self.sep_token, 4)

    def _load_vocab(self, path: str) -> Dict[str, int]:
        print("加载的 vocab 文件路径:", path)

        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return vocab

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def tokenize(self, text: Union[str, List[str]]) -> List[str]:
        if isinstance(text, str):
            return text.strip().split()
        elif isinstance(text, list):
            return text
        else:
            raise TypeError("Input must be str or List[str]")

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.vocab.get(t, self.unk_token_id) for t in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.id_to_token.get(i, self.unk_token) for i in ids]

    def encode(self, tokens: Union[str, List[str]], add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        if isinstance(tokens, str):
            tokens = self.tokenize(tokens)

        if add_special_tokens:
            tokens = [self.cls_token] + tokens + [self.sep_token]

        # 截断
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            if add_special_tokens:
                tokens[-1] = self.sep_token

        input_ids = self.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        # 填充
        padding_len = self.max_length - len(input_ids)
        if padding_len > 0:
            input_ids += [self.pad_token_id] * padding_len
            attention_mask += [0] * padding_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }

    def batch_encode(self, batch_tokens: List[Union[str, List[str]]], add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        all_input_ids = []
        all_attention_mask = []

        for tokens in batch_tokens:
            encoded = self.encode(tokens, add_special_tokens)
            all_input_ids.append(encoded["input_ids"])
            all_attention_mask.append(encoded["attention_mask"])

        return {
            "input_ids": torch.stack(all_input_ids),
            "attention_mask": torch.stack(all_attention_mask)
        }

    def decode(self, input_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()

        tokens = self.convert_ids_to_tokens(input_ids)

        if skip_special_tokens:
            special_ids = {self.pad_token_id, self.unk_token_id, self.cls_token_id, self.sep_token_id}
            tokens = [t for i, t in zip(input_ids, tokens) if i not in special_ids]

        return " ".join(tokens)
