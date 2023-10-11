import torch
from fairseq.data import LanguagePairDataset

def collate(
    samples,
    pad_idx,
    eos_idx,
    sep_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            sep_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )
    
    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
    else:
        ntokens = src_lengths.sum().item()
    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "target": target,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        )
    return batch

def collate_tokens(
    values,
    pad_idx,
    sep_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    def get_sep_posi(value):
        mx = (value == sep_idx).nonzero()
        return len(value) if len(mx) == 0 else mx[0]

    # size = max(v.size(0) for v in values)
    # size = size if pad_to_length is None else max(size, pad_to_length)
    # if pad_to_multiple != 1 and size % pad_to_multiple != 0:
    #     size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    src_size = max(get_sep_posi(v) for v in values)
    cons_size = max(len(v) - get_sep_posi(v) for v in values)
    size = src_size + cons_size

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        if cons_size > 0:
            sep_pos = get_sep_posi(v)
            if sep_pos < len(v):
                v_src = v[:sep_pos]
                v_cons = v[sep_pos:]
                copy_tensor(v_src, res[i][:len(v_src)])
                copy_tensor(v_cons, res[i][src_size: src_size + len(v_cons)])
            else:
                copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][: len(v)])
        else:
            copy_tensor(v, res[i][src_size - len(v):] if left_pad else res[i][: len(v)])
    return res

class LecaLanguagePairDataset(LanguagePairDataset):
    def __init__(
            self,
            src,
            src_sizes,
            src_dict,
            tgt=None,
            tgt_sizes=None,
            tgt_dict=None,
            left_pad_source=True,
            left_pad_target=False,
            shuffle=True,
            input_feeding=True,
            remove_eos_from_source=False,
            append_eos_to_target=False,
            align_dataset=None,
            constraints=None,
            append_bos=False,
            eos=None,
            num_buckets=0,
            src_lang_id=None,
            tgt_lang_id=None,
            pad_to_multiple=1
        ):
        super().__init__(
            src, src_sizes, src_dict, tgt, tgt_sizes, tgt_dict, left_pad_source, left_pad_target, shuffle, input_feeding,
            remove_eos_from_source, append_eos_to_target, align_dataset, constraints, append_bos, eos, num_buckets,
            src_lang_id, tgt_lang_id, pad_to_multiple
        )
        self.sep = src_dict.index("<sep>")

    def collater(self, samples, pad_to_length=None):
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            sep_idx=self.sep,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        return res