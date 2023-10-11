import logging
from .utils import post_editing, edit_distance

logger = logging.getLogger("policy")
logger.setLevel(logging.DEBUG)

class Policy():
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def initialize_episode(self):
        pass

    def revise(self, hyp, tgt):
        pass

    def encode(self, sentence: str) -> list:
        return self.tokenizer.encode(sentence)

    def decode(self, tokens: list) -> str:
        return self.tokenizer.decode(tokens)

    def max_turn(self, hypo, tgt):
        hypo_words = self.encode(hypo)
        tgt_words = self.encode(tgt)
        return edit_distance(hypo_words, tgt_words) + 1

    def accept(self, hypo, tgt):
        tgt = self.decode(self.encode(tgt)) # normalize the target
        return hypo == tgt

    def post_editing(self, hypo, tgt):
        hypo_words = self.encode(hypo)
        tgt_words = self.encode(tgt)
        return post_editing(hypo_words, tgt_words)

    def consistency(self, prev_hypo, new_hypo):
        prev = self.encode(prev_hypo) if prev_hypo else []
        new = self.encode(new_hypo)
        return edit_distance(prev, new)