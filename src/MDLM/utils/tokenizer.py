import torch
import numpy as np
from . import vocab as V
from ..data import Protein, ProteinTensorBatch, MaskedProteinTensorBatch


class ProteinTokenizer():
    def __init__(self):
        self.seq_dict = {k: v for v, k in enumerate(V.SEQUENCE_VOCAB)}
        self.vdj_dict = {k: v for v, k in enumerate(V.VDJ_VOCAB)}
        self.anarci_dict = {k: v for v, k in enumerate(V.ANARCI_VOCAB)}
        self.mask_token = V.MASK_STR
        self.mask_token_id = V.MASK_TOKEN
        self.vocab_size = len(self.seq_dict)

    def pad(
        self, 
        data: Protein, 
        max_len: int
    ) -> Protein:
        seq = data.sequence + ([V.PAD_STR] * (max_len - len(data.sequence) - 2))
        vdj = data.vdj_label + [V.PAD_STR] * (max_len - len(data.vdj_label) - 2)
        anarci = data.anarci_label + [V.PAD_STR] * (max_len - len(data.anarci_label) - 2)
        return Protein(seq, vdj, anarci)

    def tokenize(
        self, 
        data: Protein, 
        to_tensor=True,
    ) -> tuple[torch.Tensor|np.ndarray, torch.Tensor|np.ndarray, torch.Tensor|np.ndarray]:
        if to_tensor:
            seq_token = torch.tensor([self.seq_dict[s] for s in data.sequence], dtype=torch.int32)
            vdj_token = torch.tensor([self.vdj_dict[s] for s in data.vdj_label], dtype=torch.int32)
            anarci_token = torch.tensor([self.anarci_dict[s] for s in data.anarci_label], dtype=torch.int32)
        else:
            seq_token = np.array([self.seq_dict[s] for s in data.sequence], dtype='int')
            vdj_token = np.array([self.vdj_dict[s] for s in data.vdj_label], dtype='int')
            anarci_token = np.array([self.anarci_dict[s] for s in data.anarci_label], dtype='int')

        return seq_token, vdj_token, anarci_token

    def detokenize(
        self, 
        seq_token: torch.Tensor, 
        vdj_token: torch.Tensor | None, 
        anarci_token: torch.Tensor | None,
        ) -> Protein:
        sequence = [V.SEQUENCE_VOCAB[int(s)] for s in seq_token] 
        vdj_label = [V.VDJ_VOCAB[int(s)] for s in vdj_token] if vdj_token is not None else []
        anarci_label = [V.ANARCI_VOCAB[int(s)] for s in anarci_token] if anarci_token is not None else []

        return Protein(
            sequence=sequence, 
            vdj_label=vdj_label, 
            anarci_label=anarci_label
        )
        
    def batch_detokenize(
        self, 
        seq_token: torch.Tensor,
        vdj_token: torch.Tensor | None = None,
        anarci_token: torch.Tensor | None = None,
    ) -> list[Protein]:
        # debatch x [n_batch, ...] -> a list of n_batch Protein objects
        seq_debatched = torch.unbind(seq_token, dim=0)
        n_batch = len(seq_debatched)
        
        if vdj_token is not None:
            vdj_debatched = torch.unbind(vdj_token, dim=0)
        else:
            vdj_debatched = [None] * n_batch
            
        if anarci_token is not None:
            anarci_debatched = torch.unbind(anarci_token, dim=0)
        else: 
            anarci_debatched = [None] * n_batch
        
        return [self.detokenize(s, v, a) for s,v,a in zip(seq_debatched, vdj_debatched, anarci_debatched)]
        
    
    def pad_tokenize(
        self, 
        data: Protein, 
        max_len: int, 
        to_tensor: bool=True
    ) -> tuple[torch.Tensor|np.ndarray, torch.Tensor|np.ndarray, torch.Tensor|np.ndarray]:
        data = self.pad(data=data, max_len=max_len)
        return self.tokenize(data, to_tensor=to_tensor)
        
        
    def mask_tokenize_pad(
        self,
        protein: Protein,
        max_len: int,
        mask_ratio: float = 0.15,
        to_tensor: bool = True,
    ) -> tuple[torch.Tensor|np.ndarray, torch.Tensor|np.ndarray, torch.Tensor|np.ndarray, torch.Tensor|np.ndarray, torch.Tensor|np.ndarray, torch.Tensor|np.ndarray]:
        seq_token, vdj_token, anarci_token = self.pad_tokenize(data=protein, max_len=max_len, to_tensor=False,)
        
        seq_mask = np.copy(seq_token)
        vdj_mask = np.copy(vdj_token)
        anarci_mask = np.copy(anarci_token)

        non_padding_idx = np.nonzero(seq_mask - V.PAD_TOKEN)[0]
        n_mask = int(len(non_padding_idx) * mask_ratio)
        mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
        seq_mask[mask_idx] = V.MASK_TOKEN
        vdj_mask[mask_idx] = V.MASK_TOKEN
        anarci_mask[mask_idx] = V.MASK_TOKEN

        if to_tensor:
            # torch embedding takes in int(int32) or long(int64)
            seq_token = torch.tensor(seq_token, dtype=torch.int32)
            vdj_token = torch.tensor(vdj_token, dtype=torch.int32)
            anarci_token = torch.tensor(anarci_token, dtype=torch.int32)

            # one_hot encoding requires int64 
            # https://stackoverflow.com/questions/56513576/converting-tensor-to-one-hot-encoded-tensor-of-indices
            seq_mask = torch.tensor(seq_mask, dtype=torch.int64)
            vdj_mask = torch.tensor(vdj_mask, dtype=torch.int64)
            anarci_mask = torch.tensor(anarci_mask, dtype=torch.int64)

        return (
            seq_token, vdj_token, anarci_token, 
            seq_mask, vdj_mask, anarci_mask
        )

    def batch_tokenize_pad(
        self, 
        data: list[Protein],
        max_len: int
    ) -> ProteinTensorBatch:
        seq_token, vdj_token, anarci_token = [], [], []
        for protein in data:
            seq_t, vdj_t, anarci_t = self.pad_tokenize(protein, max_len)
            seq_token.append(seq_t)
            vdj_token.append(vdj_t)
            anarci_token.append(anarci_t)

        return ProteinTensorBatch(
            sequence=seq_token,
            vdj_label=vdj_token,
            anarci_label=anarci_token
        )


    def batch_mask_tokenize_pad(
        self, 
        data: list[Protein],
        max_len: int,
        mask_ratio: float = 0.15,
    ) -> MaskedProteinTensorBatch:
        seq_token, vdj_token, anarci_token = [], [], []
        seq_mask, vdj_mask, anarci_mask = [], [], []
        for protein in data:
            seq_t, vdj_t, anarci_t, seq_m, vdj_m, anarci_m = self.mask_tokenize_pad(protein,  max_len=max_len, mask_ratio= mask_ratio)
            seq_token.append(seq_t)
            vdj_token.append(vdj_t)
            anarci_token.append(anarci_t)
            seq_mask.append(seq_m)
            vdj_mask.append(vdj_m)
            anarci_mask.append(anarci_m)

        return MaskedProteinTensorBatch(
            sequence=seq_token,
            vdj_label=vdj_token,
            anarci_label=anarci_token,
            sequence_mask=seq_mask,
            vdj_mask=vdj_mask,
            anarci_mask=anarci_mask
        )

    


