from .policy import Policy, logger
from imt_environment.template import Template
from .utils import char_tag

class Left2RightPolicy(Policy):
    def __init__(self, tokenizer, n) -> None:
        super().__init__(tokenizer)
        self.n = n

    def initialize_episode(self):
        self.real_n = self.n
        self.prev_wrong_pos = -self.n
        self.prefix = None

    def revise(self, hyp, tgt):
        hyp_words = self.encode(hyp)
        hyp_len = len(hyp_words)
        tgt_words = self.encode(tgt)
        tgt_len = len(tgt_words)
        failed = False

        if self.prefix is not None:
            hypothesis = "".join(hyp_words)
            if hypothesis.find(self.prefix) != 0:
                logger.warning("Prefix is not satisfied!")
                failed = True
                return None, 0, failed

        wrong_pos = min(hyp_len, tgt_len)
        for i in range(wrong_pos):
            if hyp_words[i] != tgt_words[i]:
                wrong_pos = i
                # tgt_word may be a prefix of hyp_word which may cause infinite loop. In this case we revise one more words.
                # if hyp_words[i].startswith(tgt_words[i]) and not hyp_words[i][len(tgt_words[i])].isalnum() and self.n == 1:
                #     real_n = 2
                break
        if wrong_pos < min(hyp_len, tgt_len):
            if wrong_pos <= self.prev_wrong_pos + self.real_n - 1:
                self.real_n = self.n + self.prev_wrong_pos + self.real_n - wrong_pos
                self.prev_wrong_pos = wrong_pos
            else:
                self.prev_wrong_pos = wrong_pos
                self.real_n = self.n
                if hyp_words[wrong_pos].startswith(tgt_words[wrong_pos]) or tgt_words[wrong_pos] == "\u2581":
                    self.real_n += 1
        else:
            self.prev_wrong_pos = wrong_pos
            self.real_n = self.n
        
        tag = []
        editing_cost = 0
        if wrong_pos == hyp_len:
            revise_end_pos = min(wrong_pos + self.real_n, tgt_len)
            revise_len = revise_end_pos - wrong_pos
            hyp_words.extend(tgt_words[wrong_pos: revise_end_pos])
            token_tag = [1] * hyp_len + [2] * revise_len
            self.prefix = "".join(tgt_words[:revise_end_pos])
            if len(token_tag) < tgt_len:
                hyp_words.append("\u2581*")
                token_tag.append(4)
            hyp_words.insert(0, "<bos>")
            tag = char_tag(hyp_words, token_tag)
            editing_cost = sum([t == 2 for t in tag])
        elif wrong_pos == tgt_len:
            # tgt is a prefix of hyp
            token_tag = [1] * tgt_len + [0] * (hyp_len - tgt_len)
            hyp_words.insert(0, "<bos>")
            tag = char_tag(hyp_words, token_tag)
            editing_cost = 1
        else:
            revise_end_pos = min(wrong_pos + self.real_n, tgt_len)
            revise_len = revise_end_pos - wrong_pos
            hyp_words = hyp_words[:wrong_pos] + tgt_words[wrong_pos: revise_end_pos] + hyp_words[wrong_pos:]
            self.prefix = "".join(tgt_words[:revise_end_pos])
            diff_n = self.real_n - self.n
            token_tag = [1] * (wrong_pos + diff_n) + [2] * (revise_len - diff_n)
            if len(token_tag) < tgt_len:
                token_tag.extend([4] * (hyp_len - wrong_pos))
            else:
                token_tag.extend([0] * (hyp_len - wrong_pos))
            hyp_words.insert(0, "<bos>")
            tag = char_tag(hyp_words, token_tag)
            editing_cost = sum([t == 2 for t in tag]) + 1

        return Template(self.decode(hyp_words), tag), editing_cost, failed

