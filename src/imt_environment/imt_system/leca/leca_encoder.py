import torch
from torch import nn
import math
from typing import Optional

from fairseq.models.transformer.transformer_base import Embedding
from fairseq.models.transformer.transformer_encoder import TransformerEncoderBase

class LecaEncoder(TransformerEncoderBase):
    def __init__(self, cfg, dictionary, embed_tokens, return_fc=False):
        super().__init__(cfg, dictionary, embed_tokens, return_fc)
        self.cons_pos_embed = ConsPosiEmb(embed_tokens.embedding_dim, self.padding_idx)
        self.seg_embed = Embedding(cfg.max_constraints_num + 1, cfg.encoder.embed_dim, cfg.max_constraints_num)
        self.sep_idx = dictionary.index("<sep>")
        self.max_constraints_num = cfg.max_constraints_num

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if self.sep_idx not in src_tokens.view(-1):
            if token_embedding is None:
                token_embedding = self.embed_tokens(src_tokens)
            x = embed = self.embed_scale * token_embedding
            if self.embed_positions is not None:
                x = embed + self.embed_positions(src_tokens)
            x += self.seg_embed(torch.zeros_like(src_tokens))
        else:
            sep_mask = (src_tokens == self.sep_idx).nonzero(as_tuple=True)
            sep_position = min(sep_mask[1])
            src_sent = src_tokens[:, :sep_position]
            src_x = self.embed_scale * self.embed_tokens(src_sent)
            src_position = self.embed_positions(src_sent)
            src_seg_emb = self.seg_embed(torch.zeros_like(src_sent))

            cons_sent = src_tokens[:, sep_position:]
            cons_sep_mask = (sep_mask[0], sep_mask[1] - sep_position)
            cons_x = self.embed_scale * self.embed_tokens(cons_sent)
            cons_position = self.cons_pos_embed(cons_sent, cons_sep_mask)
            cons_seg = torch.cumsum((cons_sent == self.sep_idx), dim=1).type_as(cons_sent)
            cons_seg[cons_sent == self.padding_idx] = torch.tensor([self.max_constraints_num]).type_as(cons_seg)
            cons_seg_emb = self.seg_embed(cons_seg)

            x = torch.cat((src_x + src_position + src_seg_emb, cons_x + cons_position + cons_seg_emb), dim=1)


        # if self.layernorm_embedding is not None:
        #     x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x

    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x)

        layer = self.layers[0]
        BT_flag = False
        NT_flag = False
        # torch version check, BT>=1.12.0 and NT>=1.13.0.dev20220613
        # internal format is '1.13.0a0+fb'
        # external format is '1.13.0.dev20220613'(cpu&gpu) for nightly or "1.11.0"(cpu) or '1.11.0+cu102'(gpu) for stable
        BT_version = False
        NT_version = False
        if "fb" in torch.__version__:
            BT_version = True
            NT_version = True
        else:
            if "+" in torch.__version__:
                torch_version = torch.__version__.split("+")[0]
            else:
                torch_version = torch.__version__

            torch_version = torch_version.split(".")
            int_version = (
                int(torch_version[0]) * 1000
                + int(torch_version[1]) * 10
                + int(torch_version[2])
            )
            if len(torch_version) == 3:
                if int_version >= 1120:
                    BT_version = True
                if int_version >= 1131:
                    NT_version = True
            elif len(torch_version) == 4:
                if int_version >= 1130:
                    BT_version = True
                # Consider _nested_tensor_from_mask_left_aligned is landed after "20220613"
                if int_version >= 1131 or (
                    int_version == 1130 and torch_version[3][3:] >= "20220613"
                ):
                    NT_version = True

        if (
            BT_version
            and x.dim() == 3
            and layer.load_to_BT
            and not layer.return_fc
            and layer.can_use_fastpath
            and not layer.training
            and not layer.ever_training
            and not layer.cfg_checkpoint_activations
        ):
            # Batch first can not be justified but needs user to make sure
            x = x.transpose(0, 1)
            # Check mask conditions for nested tensor
            if NT_version:
                if (
                    encoder_padding_mask is not None
                    and torch._nested_tensor_from_mask_left_aligned(
                        x, encoder_padding_mask.logical_not()
                    )
                ):
                    if not torch.is_grad_enabled() or not x.requires_grad:
                        x = torch._nested_tensor_from_mask(
                            x, encoder_padding_mask.logical_not()
                        )
                        NT_flag = True
            BT_flag = True

        # encoder layers
        if NT_flag:
            processing_mask = None
        else:
            processing_mask = encoder_padding_mask
        encoder_padding_mask_out = processing_mask if has_pads else None
        for layer in self.layers:
            lr = layer(
                x, encoder_padding_mask=encoder_padding_mask_out
            )

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)

        if NT_flag:
            x = x.to_padded_tensor(0.0)

        if NT_flag or BT_flag:
            x = x.transpose(0, 1)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [src_tokens],
            "src_lengths": [src_lengths],
        }

class ConsPosiEmb(nn.Module):
    def __init__(self, embedding_dim, padding_idx, init_size=50):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = ConsPosiEmb.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_cons, embedding_dim, padding_idx=None, startpos=1024):
        """Build sinusoidal embeddings for constraints.
        input: num_cons, number of constraints. 
        embedding_dim, dimension of embeddings
        startpos: start position of constraints, to differentiate the position of constraint from the normal src words.
        """   
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(startpos, startpos + num_cons, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_cons, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_cons, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, cons_sep_mask, startpos=1024):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = ConsPosiEmb.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
                startpos=startpos,
            )
        self.weights = self.weights.to(self._float_tensor)
        positions = self.get_positions(input, self.padding_idx, cons_sep_mask)    
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def get_positions(self, tensor, padding_idx, cons_sep_mask):
        """Replace non-padding symbols with their position numbers.
        Position numbers begin at padding_idx+1. use constraint-right-pad-source=True
        padding_idx position=1, sep_idx position = 2 , others begin with 3, 
        a little different from the figure 2 in paper.
        """
        zero_idx = (cons_sep_mask[1] == 0).nonzero(as_tuple=True)
        diff = cons_sep_mask[1].clone()
        diff[1:] -= cons_sep_mask[1][:-1]
        diff[zero_idx] = 0
        pad_mask = tensor.ne(padding_idx).int()
        cons_pos = pad_mask.clone()
        cons_pos[cons_sep_mask] -= diff
        cons_pos = (torch.cumsum(cons_pos, dim=1).type_as(cons_pos) * pad_mask).long() + 1

        # sep_cons = torch.ones_like(tensor)
        # bsz, clen = tensor.size()       
        # for b in range(bsz):
        #     for j in range(clen):
        #         if tensor[b, j] == padding_idx:
        #             break 
        #         sep_cons[b, j] = 2 if tensor[b, j] == sep_id else sep_cons[b, j - 1] + 1 
        return cons_pos