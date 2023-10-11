import random
from .left2right_infilling import Left2RightInfillingPolicy, logger

class RandomInfillingPolicy(Left2RightInfillingPolicy):
    def __init__(self, tokenizer, seed=1) -> None:
        super().__init__(tokenizer)
        random.seed(seed)

    def initialize_episode(self):
        self.constraint_spans = None
        self.tolerance = 3

    def revise_token(self, hypo_constraint_spans, revised_sent, token_tag, tgt_words):
        tgt_len = len(tgt_words)
        if self.constraint_spans and self.constraint_spans[0][0] == 0 and self.constraint_spans[0][1] == tgt_len:
            return 0
        if self.constraint_spans:
            num_blank = len(self.constraint_spans) + 1 - int(self.constraint_spans[0][0] == 0) - int(self.constraint_spans[-1][1] == tgt_len)
        else:
            num_blank = 1
        revise_pos = random.randint(0, 2 * num_blank - 1)
        logger.debug("revise_pos: {}".format(revise_pos))
        if revise_pos % 2 == 0:
            revise_pos = revise_pos // 2
            if not self.constraint_spans or (revise_pos == 0 and self.constraint_spans[0][0] > 0):
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
            else:
                revise_pos -= 1 if self.constraint_spans[0][0] > 0 else 0
                if tgt_words[self.constraint_spans[revise_pos][1]] != "\u2581" or (
                    len(self.constraint_spans) > revise_pos + 1 and 
                    self.constraint_spans[revise_pos + 1][0] < self.constraint_spans[revise_pos][1] + 2
                ):
                    revised_sent.insert(hypo_constraint_spans[revise_pos][1], tgt_words[self.constraint_spans[revise_pos][1]])
                    token_tag.insert(hypo_constraint_spans[revise_pos][1], 2)
                    editing_cost = len(tgt_words[self.constraint_spans[revise_pos][1]])
                    self.constraint_spans[revise_pos][1] += 1
                    hypo_constraint_spans[revise_pos][1] += 1
                    self.update_hypo_constraint_span(hypo_constraint_spans, revise_pos + 1)
                else:
                    revised_sent.insert(hypo_constraint_spans[revise_pos][1], tgt_words[self.constraint_spans[revise_pos][1]])
                    revised_sent.insert(hypo_constraint_spans[revise_pos][1] + 1, tgt_words[self.constraint_spans[revise_pos][1] + 1])
                    token_tag.insert(hypo_constraint_spans[revise_pos][1], 2)
                    token_tag.insert(hypo_constraint_spans[revise_pos][1] + 1, 2)
                    editing_cost = len(tgt_words[self.constraint_spans[revise_pos][1] + 1]) + 1
                    self.constraint_spans[revise_pos][1] += 2
                    hypo_constraint_spans[revise_pos][1] += 2
                    self.update_hypo_constraint_span(hypo_constraint_spans, revise_pos + 1, n=2)
        else:
            revise_pos = (revise_pos - 1) // 2
            if not self.constraint_spans or (revise_pos == num_blank - 1 and self.constraint_spans[-1][1] < tgt_len):
                self.constraint_spans.append([tgt_len - 1, tgt_len])
                hypo_constraint_spans.append([len(revised_sent), len(revised_sent) + 1])
                revised_sent.append(tgt_words[-1])
                token_tag.append(2)
                editing_cost = len(tgt_words[-1])
            else:
                revise_pos += 1 if self.constraint_spans[0][0] == 0 else 0
                if tgt_words[self.constraint_spans[revise_pos][0] - 1] != "\u2581" or (
                    (revise_pos > 0 and self.constraint_spans[revise_pos][0] < self.constraint_spans[revise_pos - 1][1] + 2)
                    or (revise_pos == 0 and self.constraint_spans[0][0] < 2)
                ):
                    revised_sent.insert(hypo_constraint_spans[revise_pos][0], tgt_words[self.constraint_spans[revise_pos][0] - 1])
                    token_tag.insert(hypo_constraint_spans[revise_pos][0], 2)
                    editing_cost = len(tgt_words[self.constraint_spans[revise_pos][0] - 1])
                    self.constraint_spans[revise_pos][0] -= 1
                    hypo_constraint_spans[revise_pos][1] += 1
                    self.update_hypo_constraint_span(hypo_constraint_spans, revise_pos + 1)
                else:
                    revised_sent.insert(hypo_constraint_spans[revise_pos][0], tgt_words[self.constraint_spans[revise_pos][0] - 1])
                    revised_sent.insert(hypo_constraint_spans[revise_pos][0], tgt_words[self.constraint_spans[revise_pos][0] - 2])
                    token_tag.insert(hypo_constraint_spans[revise_pos][0], 2)
                    token_tag.insert(hypo_constraint_spans[revise_pos][0], 2)
                    editing_cost = len(tgt_words[self.constraint_spans[revise_pos][0] - 2]) + 1
                    self.constraint_spans[revise_pos][0] -= 2
                    hypo_constraint_spans[revise_pos][1] += 2
                    self.update_hypo_constraint_span(hypo_constraint_spans, revise_pos + 1, n=2)
        return editing_cost