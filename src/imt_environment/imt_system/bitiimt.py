from fairseq_cli.generate import get_symbols_to_strip_from_output
from .imt_system import IMTSystem, logger

class Bitiimt(IMTSystem):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.src = None
        self.max_blank = 6

    def translate(self, src, template=None):
        if src != self.src:
            self.src = src
            self.encoded_src = self.encode_fn(src)
        if template is None:
            template_str = " <sep> <blank> <eob>"
        else:
            template_str = " <sep>"
            revised_hypo = template.revised_hypo[5:] if template.revised_hypo.startswith("<bos>") else template.revised_hypo
            tag = template.tag
            constraint = ""
            blank = False
            blank_num = 0
            for i in range(len(tag)):
                if tag[i] == 0:
                    continue
                elif tag[i] < 4:
                    if blank:
                        template_str += " <blank> <eob>"
                        blank_num += 1
                        blank = False
                        if blank_num >= self.max_blank:
                            constraint = "".join([revised_hypo[j] for j in range(i, len(tag)) if 0 < tag[j] < 4])
                            break
                    constraint += revised_hypo[i]
                else:
                    blank = True
                    if constraint:
                        template_str += " " + self.encode_constraint(constraint)
                        constraint = ""
            if constraint:
                template_str += " " + self.encode_constraint(constraint)
            elif blank:
                template_str += " <blank> <eob>"
            else:
                logger.warning("template error!")

        input = self.encoded_src + template_str
        logger.debug("template_str:\n{}".format(template_str))
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
        logger.debug("output:\n{}".format(hypo_str))

        hypo = []
        j = 0
        hypo_tokens = hypo_str.split(" ")
        hypo_len = len(hypo_tokens)
        template_tokens = template_str.lstrip().split(" ")[1:]
        for t in template_tokens:
            if t != "<blank>" and t != "<eob>":
                hypo.append(t)
            elif t == "<blank>":
                if j < hypo_len:
                    while j < hypo_len and hypo_tokens[j] != "<eob>":
                        hypo.append(hypo_tokens[j])
                        j += 1
                    j += 1
                else:
                    logger.warning("blank is not infilled!")
        if j < hypo_len:
            logger.warning("generate more than needed!")
        hypo_str = " ".join(hypo)
        detok_hypo_str = self.decode_fn(hypo_str)
        logger.debug("detok str:\n{}".format(detok_hypo_str))
        return detok_hypo_str

    def make_batches(self, input):
        tokens, lengths = self.task.get_interactive_tokens_and_lengths([input], lambda x: x)
        dataset = self.task.build_dataset_for_inference(tokens, lengths)
        sample = dataset[0]
        batch = dataset.collater([sample])
        return batch
