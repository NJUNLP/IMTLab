from typing import List
import numpy as np
import editdistance
from sentencepiece import SentencePieceProcessor

def edit_distance(src: List[str], tgt: List[str]) -> int:
    return editdistance.eval(src, tgt)

def edit_operations(src: List[str], tgt: List[str]):
    src_len = len(src)
    tgt_len = len(tgt)
    dis = np.ones((src_len + 1, tgt_len + 1), dtype=int) * 1000
    dis[:, 0] = np.arange(src_len + 1)
    dis[0, :] = np.arange(tgt_len + 1)
    op = np.zeros_like(dis) # 0: keep 1: insert 2: delete 3: replace
    op[0, 1:] = np.ones(tgt_len, dtype=int)
    op[1:, 0] = np.ones(src_len, dtype=int) * 2
    for i in range(1, src_len + 1):
        for j in range(1, tgt_len + 1):
            diagonal = dis[i - 1, j - 1] + int(src[i - 1] != tgt[j - 1])
            dis[i, j] = min(dis[i - 1, j] + 1, dis[i, j - 1] + 1, diagonal)
            if dis[i, j] == dis[i, j - 1] + 1:
                op[i, j] = 1
            elif dis[i, j] == diagonal > dis[i - 1, j - 1]:
                op[i, j] = 3
            elif dis[i, j] == dis[i - 1, j] + 1:
                op[i, j] = 2
    i, j = src_len, tgt_len
    op_list = []
    while i > 0 or j > 0:
        if op[i, j] == 0:
            i -= 1
            j -= 1
            op_list.insert(0, ("keep", i, j))
        elif op[i, j] == 1:
            j -= 1
            op_list.insert(0, ("ins", i, j))
        elif op[i, j] == 2:
            i -= 1
            op_list.insert(0, ("del", i, j))
        else:
            i -= 1
            j -= 1
            op_list.insert(0, ("repl", i, j))
    return op_list, dis

def post_editing(src: List[str], tgt: List[str]):
    ops, _ = edit_operations(src, tgt)
    revised_sent, token_tag = [], []
    src_pos, editing_cost = 0, 0
    for op in ops:
        if op[0] == "keep":
            revised_sent.append(src[src_pos])
            token_tag.append(1)
            src_pos += 1
        elif op[0] == "ins":
            revised_sent.append(tgt[op[2]])
            token_tag.append(2)
            editing_cost += len(tgt[op[2]])
        elif op[0] == "del":
            revised_sent.append(src[src_pos])
            token_tag.append(0)
            editing_cost += 1
            src_pos += 1
        elif op[0] == "repl":
            revised_sent.append(tgt[op[2]])
            token_tag.append(3)
            editing_cost += len(tgt[op[2]]) + 1
            src_pos += 1
    return revised_sent, token_tag, editing_cost

def char_tag(hypo, token_tag):
    tag = []
    bos = False
    if hypo[0] == "<bos>":
        hypo = hypo[1:]
        bos = True
    for w, t in zip(hypo, token_tag):
        tag += [t] * len(w)
    if not bos and hypo[0].startswith("\u2581"):
        tag.pop(0)
    return tag

def generate_template(src: List[str], tgt: List[str]):
    ops, _ = edit_operations(src, tgt)
    hypo_keep, tgt_keep = [], []
    editing_cost = 0
    token_tag = [4] * len(src)
    for op in ops:
        if op[0] == "keep":
            hypo_keep.append(op[1])
            tgt_keep.append(op[2])
            token_tag[op[1]] = 1
    tgt_constraint_spans, hypo_constraint_spans = [], []
    if tgt_keep:
        if hypo_keep[0] > 0 or tgt_keep[0] > 0:
            editing_cost += 1
        if (hypo_keep[-1] < len(src) - 1) or (tgt_keep[-1] < len(tgt) - 1):
            editing_cost += 1
        tgt_begin_idx = tgt_keep[0]
        hypo_begin_idx = hypo_keep[0]
        for i in range(1, len(tgt_keep)):
            if tgt_keep[i] - tgt_keep[i - 1] > 1:
                tgt_constraint_spans.append([tgt_begin_idx, tgt_keep[i - 1] + 1])
                hypo_constraint_spans.append([hypo_begin_idx, hypo_keep[i - 1] + 1])
                tgt_begin_idx = tgt_keep[i]
                hypo_begin_idx = hypo_keep[i]
                editing_cost += 1
            elif hypo_keep[i] - hypo_keep[i - 1] > 1:
                token_tag[hypo_keep[i - 1] + 1: hypo_keep[i]] = [0] * (hypo_keep[i] - hypo_keep[i - 1] - 1)
                editing_cost += 1
        tgt_constraint_spans.append([tgt_begin_idx, tgt_keep[-1] + 1])
        hypo_constraint_spans.append([hypo_begin_idx, hypo_keep[-1] + 1])
    else:
        editing_cost = 1
    return tgt_constraint_spans, hypo_constraint_spans, token_tag, editing_cost

class SpaceTokenizer():
    def encode(self, x: str) -> list:
        tokens = x.split(" ")
        return ["\u2581" + t for t in tokens]

    def decode(self, x: list) -> str:
        return "".join(x).replace("\u2581", " ").strip()

class SentencePieceTokenizer():
    def __init__(self, spm_model) -> None:
        self.sp = SentencePieceProcessor(model_file=spm_model)

    def encode(self, x: str) -> list:
        return self.sp.encode(x, out_type=str)

    def decode(self, x: list) -> str:
        return self.sp.decode(x)

class ZhTokenizer():
    def __init__(self, spm_model) -> None:
        self.sp = SentencePieceProcessor(model_file=spm_model)

    def encode(self, x: str) -> list:
        l = []
        i = 0
        x = self.sp.decode(self.sp.encode(x, out_type=str)) # normalize
        while i < len(x):
            if self.is_chinese(x[i]):
                c = x[i]
                if i == 0 or x[i - 1] == " ":
                    c = "\u2581" + c
                l.append(c)
                i += 1
            elif x[i].isalnum():
                j = i + 1
                alnum = x[i]
                while j < len(x) and not self.is_chinese(x[j]) and (x[j].isalnum() or x[j] == "%"):
                    alnum += x[j]
                    j += 1
                if i == 0 or x[i - 1] == " ":
                    alnum = "\u2581" + alnum
                l.append(alnum)
                i = j
            elif x[i] == " ":
                i += 1
            else:
                c = x[i]
                if i == 0 or x[i - 1] == " ":
                    c = "\u2581" + c
                l.append(c)
                i += 1
        return l

    def decode(self, x: list) -> str:
        return "".join(x).replace("\u2581", " ").strip()

    @staticmethod
    def is_chinese(c):
        if "\u4e00" <= c <= "\u9fcb":
            return True
        else:
            return False
