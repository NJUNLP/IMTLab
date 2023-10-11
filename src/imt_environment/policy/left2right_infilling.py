import numpy as np
from .policy import Policy, logger
from imt_environment.template import Template
from .utils import char_tag, generate_template

class Left2RightInfillingPolicy(Policy):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)

    def initialize_episode(self):
        self.constraint_spans = None
        self.tolerance = 3

    def revise(self, hyp, tgt):
        hyp_words = self.encode(hyp)
        tgt_words = self.encode(tgt)
        tgt_len = len(tgt_words)

        if self.constraint_spans is None:
            revised_sent, token_tag, editing_cost = self.first_revise(hyp_words, tgt_words)
        else:
            editing_cost, cons_end_pos = 0, 0
            hypo_constraint_spans = []
            token_tag = [4] * len(hyp_words)
            hypothesis = "".join(hyp_words)
            hyp_offset = np.cumsum([0] + [len(w) for w in hyp_words])
            revised_sent = hyp_words[:]
            revised = False
            for i, span in enumerate(self.constraint_spans):
                constraint = "".join(tgt_words[span[0]: span[1]])
                cons_begin_pos = hypothesis.find(constraint, cons_end_pos)
                if cons_begin_pos >= 0:
                    cons_end_pos = cons_begin_pos + len(constraint)
                    if cons_begin_pos in hyp_offset and cons_end_pos in hyp_offset:
                        cons_begin_idx = np.argwhere(hyp_offset == cons_begin_pos).item()
                        cons_end_idx = np.argwhere(hyp_offset == cons_end_pos).item()
                        hypo_constraint_spans.append([cons_begin_idx, cons_end_idx])
                        token_tag[cons_begin_idx: cons_end_idx] = [1] * (cons_end_idx - cons_begin_idx)
                    else:
                        logger.debug("tokenization error")
                        if cons_begin_pos not in hyp_offset:
                            if span[0] > 0:
                                span[0] -= 1
                                direction = "left"
                            else:
                                span[1] += 1
                                direction = "right"
                        elif cons_end_pos not in hyp_offset:
                            if span[1] < tgt_len:
                                span[1] += 1
                                direction = "right"
                            else:
                                span[0] -= 1
                                direction = "left"
                        cons_begin_idx = np.searchsorted(hyp_offset, cons_begin_pos, side="right") - 1
                        cons_end_idx = np.searchsorted(hyp_offset, cons_end_pos)
                        if direction == "left":
                            if i > 0 and span[0] < self.constraint_spans[i - 1][1]:
                                span[0] += 1
                            elif tgt_words[span[0]] == "\u2581" and ((i > 0 and span[0] > self.constraint_spans[i - 1][1]) or (i == 0 and span[0] > 0)):
                                span[0] -= 1
                        elif direction == "right":
                            if tgt_words[span[1] - 1] == "\u2581" and self.constraint_spans[i + 1][0] > span[1]:
                                span[1] += 1
                            cons_end_idx -= 1
                        revised_sent[cons_begin_idx: cons_end_idx] = tgt_words[span[0]: span[1]]
                        token_tag[cons_begin_idx: cons_end_idx] = [3] * (span[1] - span[0])
                        new_cons_len = sum(len(w) for w in tgt_words[span[0]: span[1]])
                        editing_cost += new_cons_len - len(constraint)
                        hypo_constraint_spans.append([cons_begin_idx, cons_begin_idx + span[1] - span[0]])
                        hypothesis = "".join(revised_sent)
                        cons_end_pos = hyp_offset[cons_begin_idx] + new_cons_len
                        hyp_offset = np.cumsum([0] + [len(w) for w in revised_sent])
                        revised = True
                else:
                    logger.warning("constraint is not in the hypothesis")
                    self.tolerance -= 1
                    if self.tolerance >= 0:
                        revised_sent, token_tag, editing_cost = self.first_revise(hyp_words, tgt_words)
                        revised_sent.insert(0, "<bos>")
                        tag = char_tag(revised_sent, token_tag)
                        return Template(self.decode(revised_sent), tag), editing_cost, False
                    else:
                        return None, 0, True

            span_idx = 0
            logger.debug("spans:\n{}\n{}".format(self.constraint_spans, hypo_constraint_spans))
            tgt_blank_spans = [[self.constraint_spans[i][1], self.constraint_spans[i + 1][0]] for i in range(len(self.constraint_spans) - 1)]
            hypo_blank_spans = [[hypo_constraint_spans[i][1], hypo_constraint_spans[i + 1][0]] for i in range(len(hypo_constraint_spans) - 1)]
            if self.constraint_spans and self.constraint_spans[0][0] > 0:
                tgt_blank_spans.insert(0, [0, self.constraint_spans[0][0]])
                hypo_blank_spans.insert(0, [0, hypo_constraint_spans[0][0]])
                span_idx = -1
            if self.constraint_spans and self.constraint_spans[-1][1] < tgt_len:
                tgt_blank_spans.append([self.constraint_spans[-1][1], tgt_len])
                hypo_blank_spans.append([hypo_constraint_spans[-1][1], len(revised_sent)])
            for hypo_span, tgt_span in zip(hypo_blank_spans, tgt_blank_spans):
                i, j = hypo_span[0], tgt_span[0]
                while i < hypo_span[1] and j < tgt_span[1] and revised_sent[i] == tgt_words[j]:
                    i += 1
                    j += 1
                token_tag[hypo_span[0]: i] = [1] * (i - hypo_span[0])
                if span_idx == -1 and j > 0:
                    self.constraint_spans.insert(0, [0, j])
                    hypo_constraint_spans.insert(0, [0, i])
                    span_idx = 0
                elif j > tgt_span[0]:
                    self.constraint_spans[span_idx][1] = j
                    hypo_constraint_spans[span_idx][1] = i
                s, t = hypo_span[1] - 1, tgt_span[1] - 1
                while s >= i and t >= j and revised_sent[s] == tgt_words[t]:
                    s -= 1
                    t -= 1
                token_tag[s + 1: hypo_span[1]] = [1] * (hypo_span[1] - s - 1)
                if tgt_span[1] == tgt_len and t < tgt_span[1] - 1:
                    self.constraint_spans.append([t + 1, tgt_len])
                    hypo_constraint_spans.append([s + 1, len(revised_sent)])
                elif t < tgt_span[1] - 1:
                    self.constraint_spans[span_idx + 1][0] = t + 1
                    hypo_constraint_spans[span_idx + 1][0] = s + 1
                span_idx += 1

            insert_idx, cost = self.merge_span(hypo_constraint_spans, token_tag, len(revised_sent), tgt_len)
            editing_cost += cost
            if not revised:
                editing_cost += self.revise_token(hypo_constraint_spans, revised_sent, token_tag, tgt_words)
                new_insert_idx, _ = self.merge_span(hypo_constraint_spans, token_tag, len(revised_sent), tgt_len)
                editing_cost -= len(insert_idx) - len(new_insert_idx)
                insert_idx = new_insert_idx

            for i in range(len(insert_idx)):
                revised_sent.insert(i + insert_idx[i], "\u2581*")
                token_tag.insert(i + insert_idx[i], 4)

        if hyp_words[0] == "\u2581" and hyp_words[1][0] == "\u2581":
            hyp_words.pop(0)
            token_tag.pop(0)
        revised_sent.insert(0, "<bos>")
        tag = char_tag(revised_sent, token_tag)

        logger.debug("revised_sent:\n{}".format(revised_sent))
        logger.debug("token_tag:\n{}".format(token_tag))
        return Template(self.decode(revised_sent), tag), editing_cost, False

    def merge_span(self, aligned_span=None, token_tag=None, hyp_len=None, tgt_len=None):
        insert_idx = []
        editing_cost = 0
        if self.constraint_spans and aligned_span is not None:
            if aligned_span[0][0] > 0 and self.constraint_spans[0][0] == 0:
                token_tag[:aligned_span[0][0]] = [0] * aligned_span[0][0]
            elif aligned_span[0][0] == 0 and self.constraint_spans[0][0] > 0:
                insert_idx.append(0)
            if aligned_span[0][0] > 0 or self.constraint_spans[0][0] > 0:
                editing_cost += 1

        i = 1
        while i < len(self.constraint_spans):
            if self.constraint_spans[i][0] <= self.constraint_spans[i - 1][1]:
                self.constraint_spans[i][0] = self.constraint_spans[i - 1][0]
                self.constraint_spans.pop(i - 1)
                if aligned_span is not None:
                    if aligned_span[i][0] > aligned_span[i - 1][1]:
                        token_tag[aligned_span[i - 1][1]: aligned_span[i][0]] = [0] * (aligned_span[i][0] - aligned_span[i - 1][1])
                        editing_cost += 1
                    aligned_span[i][0] = aligned_span[i - 1][0]
                    aligned_span.pop(i - 1)
            else:
                if aligned_span is not None and aligned_span[i][0] <= aligned_span[i - 1][1]:
                    insert_idx.append(aligned_span[i - 1][1])
                editing_cost += 1
                i += 1
        
        if self.constraint_spans and aligned_span is not None:
            if aligned_span[-1][1] < hyp_len and self.constraint_spans[-1][1] == tgt_len:
                token_tag[aligned_span[-1][1]:] = [0] * (hyp_len - aligned_span[-1][1])
            elif aligned_span[-1][1] == hyp_len and self.constraint_spans[-1][1] < tgt_len:
                insert_idx.append(hyp_len)
            if aligned_span[-1][1] < hyp_len or self.constraint_spans[-1][1] < tgt_len:
                editing_cost += 1
        return insert_idx, editing_cost

    def revise_token(self, hypo_constraint_spans, revised_sent, token_tag, tgt_words):
        if not self.constraint_spans or self.constraint_spans[0][0] > 0:
            if tgt_words[0] != "\u2581" or (self.constraint_spans and self.constraint_spans[0][0] < 2):
                self.constraint_spans.insert(0, [0, 1])
                hypo_constraint_spans.insert(0, [0, 1])
                self.update_hypo_constraint_span(hypo_constraint_spans, 1)
                revised_sent.insert(0, tgt_words[0])
                token_tag.insert(0, 2)
                editing_cost = len(tgt_words[0])
            else:
                self.constraint_spans.insert(0, [0, 2])
                hypo_constraint_spans.insert(0, [0, 2])
                self.update_hypo_constraint_span(hypo_constraint_spans, 1, n=2)
                revised_sent.insert(0, tgt_words[0])
                revised_sent.insert(1, tgt_words[1])
                token_tag.insert(0, 2)
                token_tag.insert(1, 2)
                editing_cost = len(tgt_words[1]) + 1
        elif self.constraint_spans[0][1] == len(tgt_words):
            editing_cost = 0
        else:
            if tgt_words[self.constraint_spans[0][1]] != "\u2581" or (
                len(self.constraint_spans) > 1 and self.constraint_spans[1][0] < self.constraint_spans[0][1] + 2
            ):
                revised_sent.insert(hypo_constraint_spans[0][1], tgt_words[self.constraint_spans[0][1]])
                token_tag.insert(hypo_constraint_spans[0][1], 2)
                editing_cost = len(tgt_words[self.constraint_spans[0][1]])
                self.constraint_spans[0][1] += 1
                hypo_constraint_spans[0][1] += 1
                self.update_hypo_constraint_span(hypo_constraint_spans, 1)
            else:
                revised_sent.insert(hypo_constraint_spans[0][1], tgt_words[self.constraint_spans[0][1]])
                revised_sent.insert(hypo_constraint_spans[0][1] + 1, tgt_words[self.constraint_spans[0][1] + 1])
                token_tag.insert(hypo_constraint_spans[0][1], 2)
                token_tag.insert(hypo_constraint_spans[0][1] + 1, 2)
                editing_cost = len(tgt_words[self.constraint_spans[0][1] + 1]) + 1
                self.constraint_spans[0][1] += 2
                hypo_constraint_spans[0][1] += 2
                self.update_hypo_constraint_span(hypo_constraint_spans, 1, n=2)
        return editing_cost

    def first_revise(self, hyp_words, tgt_words):
        tgt_len = len(tgt_words)
        self.constraint_spans, hypo_constraint_spans, token_tag, editing_cost = generate_template(hyp_words, tgt_words)
        revised_sent = hyp_words[:]
        editing_cost += self.revise_token(hypo_constraint_spans, revised_sent, token_tag, tgt_words)
        insert_idx, _ = self.merge_span(hypo_constraint_spans, token_tag, len(revised_sent), tgt_len)

        for i in range(len(insert_idx)):
            revised_sent.insert(i + insert_idx[i], "\u2581*")
            token_tag.insert(i + insert_idx[i], 4)
        return revised_sent, token_tag, editing_cost

    @staticmethod
    def update_hypo_constraint_span(span, begin, n=1):
        for i in range(begin, len(span)):
            span[i][0] += n
            span[i][1] += n