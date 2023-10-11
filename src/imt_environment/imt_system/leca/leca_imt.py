from fairseq_cli.generate import get_symbols_to_strip_from_output
from ..imt_system import IMTSystem, logger

class LecaImt(IMTSystem):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.src = None

    def translate(self, src, template=None):
        if src != self.src:
            self.src = src
            self.encoded_src = self.encode_fn(src)
        template_str = ""
        if template is not None:
            constraints = template.get_constraints()
            if len(constraints) > 10:
                constraints = constraints[:10]
                logger.warning("constraints num exceed limit!")
            logger.debug("constraints:\n{}".format(constraints))
            for cons in constraints:
                template_str += " <sep> " + self.encode_constraint(cons)
        input = self.encoded_src + template_str
        logger.debug("input:\n{}".format(input))
        batch = self.make_batches(input)
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]

        if self.use_cuda:
            src_tokens = src_tokens.cuda()
            src_lengths = src_lengths.cuda()
        
        sample = {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
            }
        }
        translation = self.task.inference_step(
            self.generator, self.models, sample
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
        tokens, lengths = self.task.get_interactive_tokens_and_lengths([input], lambda x: x)
        dataset = self.task.build_dataset_for_inference(tokens, lengths)
        sample = dataset[0]
        batch = dataset.collater([sample])
        return batch