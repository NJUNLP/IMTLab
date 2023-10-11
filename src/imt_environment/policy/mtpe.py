from .policy import Policy
from imt_environment.template import Template
from .utils import char_tag

class MtpePolicy(Policy):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
    
    def revise(self, hyp, tgt):
        revised_hypo, token_tag, editing_cost = self.post_editing(hyp, tgt)
        if revised_hypo[0] == "\u2581" and revised_hypo[1][0] == "\u2581":
            revised_hypo.pop(0)
            token_tag.pop(0)
        tag = char_tag(revised_hypo, token_tag)
        return Template(self.decode(revised_hypo), tag), editing_cost, False