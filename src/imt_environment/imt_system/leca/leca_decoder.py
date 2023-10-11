from typing import Optional, Dict, List
from torch import Tensor, nn
import torch

from fairseq.models.transformer.transformer_decoder import TransformerDecoderBase, Linear
from fairseq import utils

class LecaDecoder(TransformerDecoderBase):
    def __init__(self, cfg, dictionary, embed_tokens, no_encoder_attn=False, output_projection=None):
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn, output_projection)
        self.ptrnet = PointerNet(cfg.encoder.embed_dim, cfg.decoder.embed_dim)
        self.sep_idx = dictionary.index("<sep>")
        self.eos_idx = dictionary.eos()

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # Prevent torchscript exporting issue for dynamic quant embedding
        prev_output_tokens = prev_output_tokens.contiguous()
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        
        if encoder_out is not None and len(encoder_out["src_tokens"]) > 0:
            src_tokens = encoder_out["src_tokens"][0]
        src_tokens = src_tokens.unsqueeze(1).expand(attn.size())
        src_masks = src_tokens.eq(self.eos_idx) | src_tokens.eq(self.padding_idx) | src_tokens.eq(self.sep_idx)
        dec_enc_attn = attn.masked_fill(src_masks, float(1e-15))
        ctx = torch.bmm(dec_enc_attn, enc.transpose(0, 1))
        gate = self.ptrnet(ctx, inner_states[-1].transpose(0, 1))

        return x, {
            "attn": [attn],
            "inner_states": inner_states,
            "dec_enc_attn": dec_enc_attn,
            "gate": gate,
            "src_tokens": src_tokens
        }

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        # if not self.use_ptrnet:
        #     if log_probs:
        #         return F.log_softmax(logits, dim=-1)
        #     else:
        #         return F.softmax(logits, dim=-1)
        gate = net_output[1]["gate"].float()
        dec_enc_attn = net_output[1]["dec_enc_attn"].float()
        src_tokens = net_output[1]["src_tokens"]
        logits = utils.softmax(logits, dim=-1)
        logits = (gate * logits).scatter_add(2, src_tokens, (1 - gate) * dec_enc_attn) + 1e-10
        return torch.log(logits)


class PointerNet(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.linear = Linear(enc_hid_dim + dec_hid_dim, 1)

    def forward(self, ctx, dec_hid):
        """
        dec_hid: bsz x tgtlen x hidsize
        c_tx : bsz x tgtlen x hidsize  
        """
        x = torch.cat((ctx, dec_hid), dim=-1)  # bsz x tgtlen x hidsize ->  bsz x tgtlen x (hidsize x 2)
        x = self.linear(x)   # bsz x tgtlen x 1
        x = torch.sigmoid(x) # bsz x tgtlen x 1 
        return x