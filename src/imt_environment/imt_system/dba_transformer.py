import torch
from fairseq_cli.generate import get_symbols_to_strip_from_output
from fairseq.token_generation_constraints import pack_constraints
from .imt_system import IMTSystem, logger

class DBATransformer(IMTSystem):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.src = None

    def translate(self, src, template=None):
        if template is not None:
            constraints = template.get_constraints()
            logger.debug("constraints:\n{}".format(constraints))
            constraint_tokens = [
                self.tgt_dict.encode_line(
                    self.encode_constraint(cons),
                    append_eos=False,
                    add_if_not_exist=False,
                ) for cons in constraints
            ]
            constraints_tensor = pack_constraints([constraint_tokens])
        else:
            constraints_tensor = torch.LongTensor()

        if src != self.src:
            self.src = src
            batch = self.make_batches(src)
            # ids = batch["id"]
            src_tokens = batch["net_input"]["src_tokens"]
            src_lengths = batch["net_input"]["src_lengths"]

            if self.use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
            
            self.sample = {
                "net_input": {
                    "src_tokens": src_tokens,
                    "src_lengths": src_lengths,
                }
            }
        if self.use_cuda:
            constraints_tensor = constraints_tensor.cuda() if constraints_tensor is not None else None

        translation = self.task.inference_step(
            self.generator, self.models, self.sample, constraints=constraints_tensor
        )[0][0]
        hypo_tokens = translation["tokens"]
        hypo_str = self.tgt_dict.string(
            hypo_tokens, self.cfg.common_eval.post_process,
            extra_symbols_to_ignore=get_symbols_to_strip_from_output(self.generator)
        )
        detok_hypo_str = self.decode_fn(hypo_str)
        logger.debug("detok str:\n{}".format(detok_hypo_str))
        return detok_hypo_str

    def make_batches(self, input):
        tokens, lengths = self.task.get_interactive_tokens_and_lengths([input], self.encode_fn)
        dataset = self.task.build_dataset_for_inference(tokens, lengths)
        sample = dataset[0]
        batch = dataset.collater([sample])
        return batch
