from dataclasses import dataclass, field
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer.transformer_config import (
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from fairseq.models.transformer.transformer_base import (
    TransformerModelBase,
)
from fairseq.models.transformer.transformer_legacy import (
    base_architecture,
    transformer_wmt_en_de_big
)
from .leca_encoder import LecaEncoder
from .leca_decoder import LecaDecoder

@dataclass
class LecaTransformerConfig(TransformerConfig):
    use_ptr: bool = field(
        default=True, metadata={"help": "set to use pointer network"}
    )
    max_constraints_num: int = field(
        default=10, metadata={"help": "maximum constrained phrases number"}
    )

@register_model("leca")
class LecaTransformer(TransformerModelBase):
    def __init__(self, args, encoder, decoder):
        cfg = LecaTransformerConfig.from_namespace(args)
        super().__init__(cfg, encoder, decoder)
        self.args = args

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        # do not set defaults so that settings defaults from various architectures still works
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=True, with_prefix=""
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            args.share_decoder_input_output_embed = True

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        if not args.share_all_embeddings:
            args.min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
        cfg = LecaTransformerConfig.from_namespace(args)
        return super().build_model(cfg, task)

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return LecaEncoder(cfg, src_dict, embed_tokens)
    
    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return LecaDecoder(cfg, tgt_dict, embed_tokens)


@register_model_architecture("leca", "leca_base")
def leca_base(args):
    base_architecture(args)
    args.use_ptr = True
    args.max_constraints_num = 10

@register_model_architecture("leca", "leca_big")
def leca_big(args):
    transformer_wmt_en_de_big(args)
    args.use_ptr = True
    args.max_constraints_num = 10