import random
import numpy as np
from .policy import Policy, logger
from imt_environment.template import Template
from .utils import edit_operations, char_tag

class RandomPolicy(Policy):
    def __init__(self, tokenizer, seed=1) -> None:
        super().__init__(tokenizer)
        random.seed(seed)

    def initialize_episode(self):
        self.constraint_span = None
        self.tolerance = 3

    def revise(self, hyp, tgt):
        hyp_words = self.encode(hyp)
        hyp_len = len(hyp_words)
        tgt_words = self.encode(tgt)
        tgt_len = len(tgt_words)
        failed = False

        editing_cost = 0
        if self.constraint_span is None:
            token_tag, editing_cost = self.first_random_revise(hyp_words, tgt_words)
        else:
            prev_constraint = "".join(tgt_words[self.constraint_span[0]: self.constraint_span[1]])
            hypothesis = "".join(hyp_words)
            cons_begin_pos = hypothesis.find(prev_constraint)
            if cons_begin_pos >= 0:
                cons_end_pos = cons_begin_pos + len(prev_constraint)
                hyp_offset = np.cumsum([0] + [len(w) for w in hyp_words])
                if cons_begin_pos in hyp_offset and cons_end_pos in hyp_offset:
                    if self.constraint_span[0] == 0:
                        revise_direction = "right"
                    elif self.constraint_span[1] == tgt_len:
                        revise_direction = "left"
                    else:
                        revise_direction = "right" if random.randint(0, 1) else "left"
                    logger.debug("direction: {}".format(revise_direction))
                    cons_begin_idx = np.argwhere(hyp_offset == cons_begin_pos).item()
                    cons_end_idx = np.argwhere(hyp_offset == cons_end_pos).item()
                    if revise_direction == "right":
                        token_tag, editing_cost = self.revise_right(cons_begin_idx, cons_end_idx, hyp_words, tgt_words)
                    else:
                        token_tag, editing_cost = self.revise_left(cons_begin_idx, cons_end_idx, hyp_words, tgt_words)
                else:
                    logger.warning("tokenization error")
                    if cons_begin_pos not in hyp_offset:
                        if self.constraint_span[0] > 0:
                            self.constraint_span[0] -= 1
                        else:
                            self.constraint_span[1] += 1
                    elif cons_end_pos not in hyp_offset:
                        if self.constraint_span[1] < tgt_len:
                            self.constraint_span[1] += 1
                        else:
                            self.constraint_span[0] -= 1
                    cons_begin_idx = np.searchsorted(hyp_offset, cons_begin_pos, side="right") - 1
                    cons_end_idx = np.searchsorted(hyp_offset, cons_end_pos)
                    hyp_words[cons_begin_idx: cons_end_idx] = tgt_words[self.constraint_span[0]: self.constraint_span[1]]
                    cons_end_idx = cons_begin_idx + self.constraint_span[1] - self.constraint_span[0]
                    token_tag = [4] * cons_begin_idx + [3] * (self.constraint_span[1] - self.constraint_span[0]) + [4] * (len(hyp_words) - cons_end_idx)
                    editing_cost += sum([len(w) for w in tgt_words[self.constraint_span[0]: self.constraint_span[1]]]) - len(prev_constraint)
                    editing_cost += self.check_boundaries(cons_begin_idx, cons_end_idx, hyp_words, token_tag, tgt_len)
            else:
                logger.warning("constraint is not in the hypothesis")
                self.tolerance -= 1
                if self.tolerance >= 0:
                    token_tag, editing_cost = self.first_random_revise(hyp_words, tgt_words)
                else:
                    failed = True
                    return None, 0, failed

        if hyp_words[0] == "\u2581" and hyp_words[1][0] == "\u2581":
            hyp_words.pop(0)
            token_tag.pop(0)
        hyp_words.insert(0, "<bos>")
        tag = char_tag(hyp_words, token_tag)

        return Template(self.decode(hyp_words), tag), editing_cost, failed

    def first_random_revise(self, hyp_words, tgt_words):
        hyp_len = len(hyp_words)
        edit_ops, _ = edit_operations(hyp_words, tgt_words)
        revise_ops = [op for op in edit_ops if op[0] in ["ins", "repl"]]
        if revise_ops:
            new_revise_ops = [op for op in revise_ops if tgt_words[op[2]] != "\u2581"]
            if new_revise_ops:
                revise_op = random.choice(new_revise_ops)
            else:
                editing_cost = 0
                token_tag = []
                for i, op in enumerate(edit_ops):
                    if op[0] == "keep":
                        token_tag.append(1)
                    elif op[0] == "ins":
                        hyp_words.insert(i, "\u2581")
                        editing_cost += 1
                        token_tag.append(2)
                    elif op[0] == "repl":
                        hyp_words[i] = "\u2581"
                        editing_cost += 2
                        token_tag.append(3)
                    elif op[0] == "del":
                        token_tag.append(0)
                        if i > 0 and edit_ops[i - 1][0] != "del":
                            editing_cost += 1
                if edit_ops[0][0] == "del":
                    editing_cost += 1
                return token_tag, editing_cost
        else:
            token_tag = [int(op[0] == "keep") for op in edit_ops]
            editing_cost = 0
            for i in range(len(token_tag) - 1):
                if token_tag[i] == 0 and token_tag[i + 1] == 1:
                    editing_cost += 1
            editing_cost += int(token_tag[-1] == 0)
            return token_tag, editing_cost

        hyp_words.insert(revise_op[1], tgt_words[revise_op[2]])
        hyp_len += 1
        token_tag = [4] * hyp_len
        token_tag[revise_op[1]] = 2
        editing_cost = len(tgt_words[revise_op[2]])

        cons_begin_idx = revise_op[1]
        cons_end_idx = revise_op[1] + 1
        self.constraint_span = [revise_op[2], revise_op[2] + 1]
        editing_cost += self.check_boundaries(cons_begin_idx, cons_end_idx, hyp_words, token_tag, len(tgt_words))

        return token_tag, editing_cost

    def revise_right(self, cons_begin_idx, cons_end_idx, hyp_words, tgt_words):
        hyp_len = len(hyp_words)
        tgt_len = len(tgt_words)
        editing_cost = 0
        i, j = cons_end_idx, self.constraint_span[1]
        while i < hyp_len and j < tgt_len and hyp_words[i] == tgt_words[j]:
            i += 1
            j += 1
        if j == tgt_len:
            self.constraint_span[1] = j
            if self.constraint_span[0] == 0:
                token_tag = [0] * cons_begin_idx + [1] * (i - cons_begin_idx) + [0] * (hyp_len - i)
                editing_cost = int(cons_begin_idx > 0) + int(hyp_len > i)
            else:
                token_tag, editing_cost = self.revise_left(cons_begin_idx, i, hyp_words, tgt_words)
        else:
            if tgt_words[j] != "\u2581":
                hyp_words.insert(i, tgt_words[j])
                self.constraint_span[1] = j + 1
                cons_end_idx = i + 1
                token_tag = [4] * cons_begin_idx + [1] * (i - cons_begin_idx) + [2] + [4] * (hyp_len - i)
                editing_cost += len(tgt_words[j])
            else:
                hyp_words.insert(i, tgt_words[j])
                hyp_words.insert(i + 1, tgt_words[j + 1])
                self.constraint_span[1] = j + 2
                cons_end_idx = i + 2
                token_tag = [4] * cons_begin_idx + [1] * (i - cons_begin_idx) + [2, 2] + [4] * (hyp_len - i)
                editing_cost += len(tgt_words[j + 1]) + 1
            editing_cost += self.check_boundaries(cons_begin_idx, cons_end_idx, hyp_words, token_tag, tgt_len)

        return token_tag, editing_cost

    def revise_left(self, cons_begin_idx, cons_end_idx, hyp_words, tgt_words):
        hyp_len = len(hyp_words)
        tgt_len = len(tgt_words)
        editing_cost = 0
        i, j = cons_begin_idx - 1, self.constraint_span[0] - 1
        while i >= 0 and j >= 0 and hyp_words[i] == tgt_words[j]:
            i -= 1
            j -= 1
        if j == -1:
            self.constraint_span[0] = 0
            if self.constraint_span[1] == tgt_len:
                token_tag = [0] * (i + 1) + [1] * (cons_end_idx - i - 1) + [0] * (hyp_len - cons_end_idx)
                editing_cost = int(i > -1) + int(cons_end_idx < hyp_len)
            else:
                token_tag, editing_cost = self.revise_right(i + 1, cons_end_idx, hyp_words, tgt_words)
        else:
            if tgt_words[j] == "\u2581" and j > 0:
                hyp_words.insert(i + 1, tgt_words[j])
                hyp_words.insert(i + 1, tgt_words[j - 1])
                self.constraint_span[0] = j - 1
                cons_begin_idx = i + 1
                cons_end_idx += 2
                token_tag = [4] * (i + 1) + [2, 2] + [1] * (cons_end_idx - i - 3) + [4] * (hyp_len - cons_end_idx + 2)
                editing_cost += len(tgt_words[j - 1]) + 1
            else:
                hyp_words.insert(i + 1, tgt_words[j])
                self.constraint_span[0] = j
                cons_begin_idx = i + 1
                cons_end_idx += 1
                token_tag = [4] * (i + 1) + [2] + [1] * (cons_end_idx - i - 2) + [4] * (hyp_len - cons_end_idx + 1)
                editing_cost += len(tgt_words[j])
            editing_cost += self.check_boundaries(cons_begin_idx, cons_end_idx, hyp_words, token_tag, tgt_len)

        return token_tag, editing_cost

    def check_boundaries(self, cons_begin, cons_end, hyp_words, token_tag, tgt_len):
        editing_cost = 0
        hyp_len = len(hyp_words)
        if self.constraint_span[1] == tgt_len and cons_end < hyp_len:
            token_tag[cons_end:] = [0] * (hyp_len - cons_end)
        elif self.constraint_span[1] < tgt_len and cons_end == hyp_len:
            hyp_words.append("\u2581*")
            token_tag.append(4)
        if self.constraint_span[1] < tgt_len or cons_end < hyp_len:
            editing_cost += 1

        if self.constraint_span[0] == 0 and cons_begin > 0:
            token_tag[:cons_begin] = [0] * cons_begin
        elif self.constraint_span[0] > 0 and cons_begin == 0:
            hyp_words.insert(0, "\u2581*")
            token_tag.insert(0, 4)
        if self.constraint_span[0] > 0 or cons_begin > 0:
            editing_cost += 1
        return editing_cost
