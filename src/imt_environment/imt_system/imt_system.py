import logging
import numpy as np
import torch
import ast

from fairseq import options, utils, tasks, checkpoint_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

logger = logging.getLogger("imt_system")
logger.setLevel(logging.DEBUG)

class IMTSystem():
    def __init__(self, args) -> None:
        parser = options.get_interactive_generation_parser()
        cfg = options.parse_args_and_arch(parser, args)
        cfg = convert_namespace_to_omegaconf(cfg)
        utils.import_user_module(cfg.common)

        logger.info(cfg)

        # Fix seed for stochastic decoding
        if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
            np.random.seed(cfg.common.seed)
            utils.set_torch_seed(cfg.common.seed)

        use_cuda = torch.cuda.is_available() and not cfg.common.cpu

        task = tasks.setup_task(cfg.task)

        # Load ensemble
        overrides = ast.literal_eval(cfg.common_eval.model_overrides)
        logger.info("loading model(s) from {}".format(cfg.common_eval.path))
        models, _model_args = checkpoint_utils.load_model_ensemble(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            task=task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )

        # Set dictionaries
        src_dict = task.source_dictionary
        tgt_dict = task.target_dictionary

        # Optimize ensemble for generation
        for model in models:
            if model is None:
                continue
            if cfg.common.fp16:
                model.half()
            if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
                model.cuda()
            model.prepare_for_inference_(cfg)
        
        # Initialize generator
        generator = task.build_generator(models, cfg.generation)

        # Handle tokenization and BPE
        tokenizer = task.build_tokenizer(cfg.tokenizer)
        bpe = task.build_bpe(cfg.bpe)

        def encode_fn(x):
            if tokenizer is not None:
                x = tokenizer.encode(x)
            if bpe is not None:
                x = bpe.encode(x)
            return x

        def decode_fn(x):
            if bpe is not None:
                x = bpe.decode(x)
            if tokenizer is not None:
                x = tokenizer.decode(x)
            return x

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        align_dict = utils.load_align_dict(cfg.generation.replace_unk)

        max_positions = utils.resolve_max_positions(
            task.max_positions(), *[model.max_positions() for model in models]
        )

        self.cfg = cfg
        self.use_cuda = use_cuda
        self.task = task
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.models = models
        self.generator = generator
        self.align_dict = align_dict
        self.max_positions = max_positions
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn

        logger.info("initialize done!")

    def encode_constraint(self, cons):
        cons_tokens = self.encode_fn(cons)
        if cons.startswith(" "):
            return cons_tokens
        else:
            if cons_tokens[1] == " ":
                return cons_tokens[2:]
            else:
                return cons_tokens[1:]

    def translate(self, src, template=None):
        pass

    def update(self, src, tgt):
        pass
