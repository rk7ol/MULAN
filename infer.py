#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ProteinGym-style zero-shot inference with MULAN.

This script aligns the zero-shot mutation scoring interface with the `esm/` and
`saport/` projects in this workspace:
  - Input CSV schema: `mutant`, `mutated_sequence`, `DMS_score`
  - Output: per-variant delta log-probability + `summary.csv`

Two scoring modes are supported:
  1) Sequence-only (ESM2 tokenizer): mask one residue and compute
       Δ = log P(mut | context) - log P(wt | context)
  2) Foldseek/SaProt tokenization (AA+3Di): mask the AA but keep the 3Di tag at
     the site, and compute a masked-marginal score marginalized over the 3Di
     sub-vocabulary (SaProt-style).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer

from mulan.foldseek_utils import get_struc_seq
from mulan.model import StructEsmForMaskedLM
from mulan.pdb_utils import AnglesFromStructure, getStructureObject
from mulan.tokenizer import Tokenizer as StructureTokenizer
from mulan.utils import get_foldseek_tokenizer

try:
    from importlib.metadata import distributions
except Exception:  # pragma: no cover

    def distributions():  # type: ignore[override]
        return []


REQUIRED_COLS = {"mutant", "mutated_sequence", "DMS_score"}
AA20 = "ACDEFGHIKLMNPQRSTVWY"
FOLDSEEK_STRUC_VOCAB = list("pynwrqhgdlvtmfsaeikc")


def compute_spearman(pred_scores, true_scores) -> tuple[float | None, float | None]:
    rho, pval = spearmanr(pred_scores, true_scores, nan_policy="omit")
    rho_val = None if rho is None or (isinstance(rho, float) and math.isnan(rho)) else float(rho)
    pval_val = None if pval is None or (isinstance(pval, float) and math.isnan(pval)) else float(pval)
    return rho_val, pval_val


def _fmt_float(x: float | None, *, fmt: str) -> str:
    return "nan" if x is None else format(x, fmt)


def collect_installed_packages() -> list[str]:
    items: list[str] = []
    for dist in distributions():
        name = None
        try:
            name = dist.metadata.get("Name")
        except Exception:
            name = None
        if not name:
            continue
        items.append(f"{name}=={dist.version}")
    return sorted(set(items), key=str.lower)


def print_runtime_environment() -> None:
    print("========== Runtime ==========")
    print(f"Python:        {sys.version.replace(os.linesep, ' ')}")
    print(f"Executable:    {sys.executable}")
    print(f"Platform:      {sys.platform}")
    print("Packages:")
    for item in collect_installed_packages():
        print(f"  - {item}")
    print("=============================\n")


def parse_mutant(mut_str: str) -> tuple[str, int, str]:
    wt_aa = mut_str[0]
    mut_aa = mut_str[-1]
    pos1 = int(mut_str[1:-1])
    return wt_aa, pos1, mut_aa


def recover_wt_sequence(mut_seq: str, wt_aa: str, pos1: int) -> str:
    return mut_seq[: pos1 - 1] + wt_aa + mut_seq[pos1:]


def resolve_csv_paths(*, data_dir: Path, csv: str | None) -> list[Path]:
    if csv:
        p = Path(csv)
        if p.exists():
            return [p]
        candidate = data_dir / csv
        if candidate.exists():
            return [candidate]
        raise FileNotFoundError(f"input_csv not found: {csv!r} (searched in {data_dir})")

    # Batch mode: all CSVs in directory
    return sorted([p for p in data_dir.glob("*.csv") if p.is_file()])


def load_dataset(*, csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} missing required columns: {sorted(missing)}")
    return df


def _token_window(tokens: list[str], center: int, window: int) -> None:
    lo = max(0, center - window)
    hi = min(len(tokens), center + window + 1)
    for i in range(lo, hi):
        mark = "<==" if i == center else "   "
        print(f"  {i:5d}: {tokens[i]!r} {mark}")


@torch.no_grad()
def debug_alignment_mulan(
    *,
    tokenizer,
    model: StructEsmForMaskedLM,
    device: str,
    mutant: str,
    masked_input: str,
    pos1: int,
    token_window_size: int,
) -> None:
    enc = tokenizer(masked_input, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids = enc["input_ids"][0].tolist()
    id_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print("\n========== Alignment Debug (MULAN) ==========")
    print(f"mutant: {mutant}")
    print(f"candidate token indices: pos1-1={pos1-1}, pos1={pos1}")
    print(f"token[0]: {id_tokens[0]!r}  (expected <cls>)")
    if 0 <= pos1 - 1 < len(id_tokens):
        print(f"token[pos1-1]: {id_tokens[pos1-1]!r}")
    if 0 <= pos1 < len(id_tokens):
        print(f"token[pos1]:   {id_tokens[pos1]!r}  (expected masked site)")
        print("token window around pos1:")
        _token_window(id_tokens, center=pos1, window=token_window_size)

    out = model(**enc)
    logits = out.logits["scores"][0, pos1, :]
    topk = torch.topk(torch.softmax(logits, dim=-1), k=10)
    top_ids = topk.indices.tolist()
    top_ps = topk.values.tolist()
    top_tokens = tokenizer.convert_ids_to_tokens(top_ids)
    print("\ntop-10 tokens:")
    for t, p in zip(top_tokens, top_ps, strict=True):
        print(f"  {t!r}\t{p:.4g}")


def _auto_base_checkpoint(*, config, use_foldseek_sequences: bool, foldseek_base: str) -> str | None:
    layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    if use_foldseek_sequences:
        if layers == 12:
            return "westlake-repl/SaProt_35M_AF2" if foldseek_base == "af2" else "westlake-repl/SaProt_35M_PDB"
        if layers == 33:
            return "westlake-repl/SaProt_650M_AF2" if foldseek_base == "af2" else "westlake-repl/SaProt_650M_PDB"
        return None

    if layers == 6:
        return "facebook/esm2_t6_8M_UR50D"
    if layers == 12:
        return "facebook/esm2_t12_35M_UR50D"
    if layers == 33:
        return "facebook/esm2_t33_650M_UR50D"
    return None


def load_model_and_tokenizers(
    *,
    model_id_or_path: str,
    use_struct_embeddings: bool,
    num_struct_embeddings_layers: int,
    mask_angle_inputs_with_plddt: bool,
    use_foldseek_sequences: bool,
    add_foldseek_embeddings: bool,
    esm_checkpoint: str | None,
    foldseek_base: str,
) -> tuple[StructEsmForMaskedLM, object, object | None, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fs_tokenizer = None
    if add_foldseek_embeddings:
        fs_tokenizer = get_foldseek_tokenizer()

    model = StructEsmForMaskedLM.from_pretrained(
        model_id_or_path,
        num_struct_embeddings_layers=num_struct_embeddings_layers,
        struct_data_dim=7,
        use_struct_embeddings=use_struct_embeddings,
        predict_contacts="none",
        predict_angles=False,
        mask_angle_inputs_with_plddt=mask_angle_inputs_with_plddt,
        add_foldseek_embeddings=add_foldseek_embeddings,
        fs_tokenizer=fs_tokenizer,
    )
    model.eval()
    model.to(device)

    base_checkpoint = esm_checkpoint or _auto_base_checkpoint(
        config=model.config, use_foldseek_sequences=use_foldseek_sequences, foldseek_base=foldseek_base
    )

    if base_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)

    return model, tokenizer, fs_tokenizer, device


def build_pdb_struct_context(
    *,
    pdb_path: Path,
    pdb_chain: str,
    pad_value: float,
) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
    structure = getStructureObject(str(pdb_path), chain=pdb_chain)
    angles_df = AnglesFromStructure(structure)

    tok = StructureTokenizer(use_foldseek_sequences=False)
    protein = tok.tokenize_protein(angles_df)
    seqs, angles_list, _names = tok.preproc_structural_data([protein], ["protein"])
    if not seqs:
        raise ValueError(f"Failed to extract any residues from structure: {pdb_path}")

    wt_seq = seqs[0]
    raw = np.asarray(angles_list[0], dtype=np.float32)  # (L, 11) = [plddt, xyz, angles...]

    L = raw.shape[0]
    tensor_batch = np.ones((1, L + 2, raw.shape[-1]), dtype=np.float32) * float(pad_value)
    tensor_batch[0, 1 : L + 1, :] = raw

    # Match ProteinDataset.form_batches: if dim==11 drop [plddt, xyz] leaving angles (7 dims).
    if tensor_batch.shape[-1] != 11:
        raise ValueError(f"Unexpected structural feature dim {tensor_batch.shape[-1]} (expected 11)")
    angles_for_model = torch.from_numpy(tensor_batch[:, :, 4:])  # (1, L+2, 7)
    plddts = torch.from_numpy(tensor_batch[:, 1:-1, 0])  # (1, L)
    coords = torch.from_numpy(tensor_batch[:, 1:-1, 1:4])  # (1, L, 3)
    return wt_seq, angles_for_model, plddts, coords


def build_foldseek_combined_seq(
    *,
    foldseek_path: Path,
    pdb_path: Path,
    pdb_chain: str,
    plddt_mask: bool | str,
    plddt_threshold: float,
) -> str:
    seq_dict = get_struc_seq(
        str(foldseek_path),
        str(pdb_path),
        chains=[pdb_chain],
        plddt_mask=bool(plddt_mask) if plddt_mask != "auto" else False,
        plddt_threshold=float(plddt_threshold),
    )
    if pdb_chain not in seq_dict:
        raise ValueError(f"Foldseek did not return chain {pdb_chain!r}; available: {sorted(seq_dict.keys())}")
    _seq, _struc_seq, combined = seq_dict[pdb_chain]
    return combined


def _make_padded_plddts(*, input_ids: torch.Tensor, plddts: torch.Tensor, pad_value: float) -> torch.Tensor:
    # Non-T5 layout: [<cls>] + residues + [<eos>]
    padded = torch.ones_like(input_ids, dtype=torch.float32) * float(pad_value)
    padded[:, 1:-1] = plddts.to(dtype=torch.float32)
    return padded


@torch.no_grad()
def score_delta_logp_seq(
    *,
    model: StructEsmForMaskedLM,
    tokenizer,
    device: str,
    batch_size: int,
    wt_seqs: list[str],
    wt_aas: list[str],
    pos1s: list[int],
    mut_aas: list[str],
    angles_for_model: torch.Tensor | None,
    plddts: torch.Tensor | None,
    pad_value: float,
    progress_every: int,
) -> list[float]:
    scores: list[float] = []

    for start in range(0, len(wt_seqs), batch_size):
        end = min(len(wt_seqs), start + batch_size)
        enc = tokenizer.batch_encode_plus(wt_seqs[start:end], padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Apply one-mask-per-sequence at token index `pos1` (since token[0] is <cls>).
        for i in range(end - start):
            input_ids[i, int(pos1s[start + i])] = int(tokenizer.mask_token_id)

        struct_inputs = None
        if angles_for_model is not None and plddts is not None:
            b = end - start
            struct_angles = angles_for_model.expand(b, -1, -1).to(device)
            padded_plddts = _make_padded_plddts(input_ids=input_ids, plddts=plddts.expand(b, -1), pad_value=pad_value).to(
                device
            )
            struct_inputs = (struct_angles, padded_plddts, None, [])

        out = model(input_ids=input_ids, attention_mask=attention_mask, struct_inputs=struct_inputs)
        logits = out.logits["scores"]  # (B, T, V)
        log_probs = torch.log_softmax(logits, dim=-1)

        for i in range(end - start):
            pos1 = int(pos1s[start + i])
            wt_id = int(tokenizer.convert_tokens_to_ids(wt_aas[start + i]))
            mut_id = int(tokenizer.convert_tokens_to_ids(mut_aas[start + i]))
            scores.append(float(log_probs[i, pos1, mut_id].item() - log_probs[i, pos1, wt_id].item()))

        if progress_every and len(scores) % int(progress_every) == 0:
            print(f"Scored {len(scores)}/{len(wt_seqs)} variants...")

    return scores


def _aa_logprob_mass_foldseek(log_probs_site: torch.Tensor, *, tokenizer, aa: str) -> torch.Tensor:
    ids: list[int] = []
    for c in FOLDSEEK_STRUC_VOCAB:
        tok = f"{aa}{c}"
        tid = int(tokenizer.convert_tokens_to_ids(tok))
        if tid < 0 or tid == getattr(tokenizer, "unk_token_id", None):
            raise ValueError(f"Tokenizer cannot resolve SaProt token: {tok!r} -> {tid}")
        ids.append(tid)
    return torch.logsumexp(log_probs_site[torch.tensor(ids, device=log_probs_site.device)], dim=0)


@torch.no_grad()
def score_delta_logp_foldseek(
    *,
    model: StructEsmForMaskedLM,
    tokenizer,
    device: str,
    batch_size: int,
    masked_seqs: list[str],
    wt_aas: list[str],
    pos1s: list[int],
    mut_aas: list[str],
    angles_for_model: torch.Tensor,
    plddts: torch.Tensor,
    pad_value: float,
    progress_every: int,
) -> list[float]:
    scores: list[float] = []

    for start in range(0, len(masked_seqs), batch_size):
        end = min(len(masked_seqs), start + batch_size)
        enc = tokenizer.batch_encode_plus(masked_seqs[start:end], padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        b = end - start
        struct_angles = angles_for_model.expand(b, -1, -1).to(device)
        padded_plddts = _make_padded_plddts(input_ids=input_ids, plddts=plddts.expand(b, -1), pad_value=pad_value).to(device)
        struct_inputs = (struct_angles, padded_plddts, None, [])

        out = model(input_ids=input_ids, attention_mask=attention_mask, struct_inputs=struct_inputs)
        logits = out.logits["scores"]
        log_probs = torch.log_softmax(logits, dim=-1)

        for i in range(end - start):
            pos1 = int(pos1s[start + i])
            lp = log_probs[i, pos1, :]
            lp_wt = _aa_logprob_mass_foldseek(lp, tokenizer=tokenizer, aa=wt_aas[start + i])
            lp_mut = _aa_logprob_mass_foldseek(lp, tokenizer=tokenizer, aa=mut_aas[start + i])
            scores.append(float((lp_mut - lp_wt).item()))

        if progress_every and len(scores) % int(progress_every) == 0:
            print(f"Scored {len(scores)}/{len(masked_seqs)} variants...")

    return scores


def run_one_csv(
    *,
    csv_path: Path,
    output_dir: Path,
    output_suffix: str,
    model: StructEsmForMaskedLM,
    tokenizer,
    device: str,
    batch_size: int,
    use_foldseek_sequences: bool,
    use_struct_embeddings: bool,
    compare_no3di: bool,
    tag: str,
    pdb_wt_seq: str | None,
    foldseek_combined_seq: str | None,
    angles_for_model: torch.Tensor | None,
    plddts: torch.Tensor | None,
    pad_value: float,
    progress_every: int,
    debug_alignment: bool,
    debug_rows: int,
    debug_token_window: int,
) -> dict | None:
    df = load_dataset(csv_path=csv_path)

    print(f"Loaded {len(df)} variants. Building masked inputs...")
    wt_aas: list[str] = []
    pos1s: list[int] = []
    mut_aas: list[str] = []
    true_scores: list[float] = []

    wt_seqs: list[str] = []
    masked_seqs: list[str] = []
    masked_seqs_no3di: list[str] = []

    for idx, row in df.iterrows():
        mutant = str(row["mutant"])
        mut_seq = str(row["mutated_sequence"])
        wt_aa, pos1, mut_aa = parse_mutant(mutant)
        wt_seq = recover_wt_sequence(mut_seq=mut_seq, wt_aa=wt_aa, pos1=pos1)

        if pos1 < 1 or pos1 > len(wt_seq):
            raise ValueError(f"Row {idx}: position out of range for sequence length {len(wt_seq)}: {mutant!r}")

        if pdb_wt_seq is not None and wt_seq != pdb_wt_seq:
            raise ValueError(
                f"Row {idx}: recovered WT sequence does not match structure WT sequence.\n"
                f"  mutant={mutant!r}\n"
                f"  len(csv_wt)={len(wt_seq)} len(pdb_wt)={len(pdb_wt_seq)}"
            )

        wt_aas.append(wt_aa)
        pos1s.append(pos1)
        mut_aas.append(mut_aa)
        true_scores.append(float(row["DMS_score"]))

        if use_foldseek_sequences:
            assert foldseek_combined_seq is not None
            aa_from_structure = foldseek_combined_seq[0::2]
            if aa_from_structure != wt_seq:
                raise ValueError(
                    f"Row {idx}: WT sequence mismatch between CSV and foldseek-derived structure sequence.\n"
                    f"  mutant={mutant!r}\n"
                    f"  len(csv_wt)={len(wt_seq)} len(struct_aa)={len(aa_from_structure)}"
                )

            chars = list(foldseek_combined_seq)
            aa_index = (pos1 - 1) * 2
            chars[aa_index] = "#"
            masked_seqs.append("".join(chars))

            if compare_no3di:
                base = []
                for a in wt_seq:
                    base.append(a)
                    base.append(tag)
                base_chars = base
                base_chars[aa_index] = "#"
                masked_seqs_no3di.append("".join(base_chars))
        else:
            wt_seqs.append(wt_seq)

        if progress_every and (idx + 1) % int(progress_every) == 0:
            print(f"Prepared {idx+1}/{len(df)} variants...")

    if use_foldseek_sequences:
        assert angles_for_model is not None and plddts is not None, "foldseek mode requires structural angles + plddts"
        pred = score_delta_logp_foldseek(
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            masked_seqs=masked_seqs,
            wt_aas=wt_aas,
            pos1s=pos1s,
            mut_aas=mut_aas,
            angles_for_model=angles_for_model,
            plddts=plddts,
            pad_value=pad_value,
            progress_every=progress_every,
        )
        df["mulan_delta_logp"] = pred
        rho, pval = compute_spearman(pred, true_scores)

        rho2 = pval2 = None
        if compare_no3di:
            pred2 = score_delta_logp_foldseek(
                model=model,
                tokenizer=tokenizer,
                device=device,
                batch_size=batch_size,
                masked_seqs=masked_seqs_no3di,
                wt_aas=wt_aas,
                pos1s=pos1s,
                mut_aas=mut_aas,
                angles_for_model=angles_for_model,
                plddts=plddts,
                pad_value=pad_value,
                progress_every=progress_every,
            )
            df["mulan_delta_logp_no3di"] = pred2
            rho2, pval2 = compute_spearman(pred2, true_scores)

    else:
        pred = score_delta_logp_seq(
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            wt_seqs=wt_seqs,
            wt_aas=wt_aas,
            pos1s=pos1s,
            mut_aas=mut_aas,
            angles_for_model=angles_for_model if use_struct_embeddings else None,
            plddts=plddts if use_struct_embeddings else None,
            pad_value=pad_value,
            progress_every=progress_every,
        )
        df["mulan_delta_logp"] = pred
        rho, pval = compute_spearman(pred, true_scores)
        rho2 = pval2 = None

    if debug_alignment and len(df) > 0:
        limit = min(len(df), max(1, int(debug_rows)))
        for i in range(limit):
            row = df.iloc[i]
            mutant = str(row["mutant"])
            wt_aa, pos1, mut_aa = parse_mutant(mutant)
            if use_foldseek_sequences:
                seq = masked_seqs[i]
            else:
                # For debug in seq mode, build a display-only masked input by masking the AA as 'X';
                # the model call in seq mode uses token-id replacement.
                wt_seq = recover_wt_sequence(str(row["mutated_sequence"]), wt_aa, pos1)
                seq = wt_seq[: pos1 - 1] + "X" + wt_seq[pos1:]
            debug_alignment_mulan(
                tokenizer=tokenizer,
                model=model,
                device=device,
                mutant=mutant,
                masked_input=seq,
                pos1=pos1,
                token_window_size=max(1, int(debug_token_window)),
            )

    out_csv = output_dir / f"{csv_path.stem}{output_suffix}"
    df.to_csv(out_csv, index=False)

    print("\n========== ProteinGym zero-shot ==========")
    print("Model:        MULAN")
    print(f"CSV:          {csv_path.name}")
    print(f"Variants:     {len(df)}")
    print(f"Spearman ρ:   {_fmt_float(rho, fmt='.4f')}")
    print(f"P-value:      {_fmt_float(pval, fmt='.2e')}")
    if compare_no3di:
        print(f"Spearman ρ*:  {_fmt_float(rho2, fmt='.4f')}  (no3Di baseline)")
        print(f"P-value*:     {_fmt_float(pval2, fmt='.2e')}")
    print(f"Saved to:     {out_csv}")
    print("==========================================\n")

    return {
        "csv": csv_path.name,
        "variants": int(len(df)),
        "spearman_rho": rho,
        "p_value": pval,
        "spearman_rho_no3di": rho2,
        "p_value_no3di": pval2,
        "output_csv": str(out_csv),
        "use_foldseek_sequences": bool(use_foldseek_sequences),
        "use_struct_embeddings": bool(use_struct_embeddings),
        "score_column": "mulan_delta_logp",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="DFrolova/MULAN-small", help="HF model id or local checkpoint path.")
    p.add_argument("--esm_checkpoint", default=None, help="Optional: override base tokenizer checkpoint.")
    p.add_argument("--foldseek_base", choices=["af2", "pdb"], default="af2", help="SaProt base model family for foldseek tokenization.")

    p.add_argument("--input_csv", default=None, help="Only process this CSV (basename under data_dir, or absolute path).")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--progress_every", type=int, default=100)
    p.add_argument("--output_suffix", default="_mulan_zeroshot.csv")

    p.add_argument("--use_foldseek_sequences", action="store_true", help="Use SaProt AA+3Di tokenization (requires pdb + foldseek).")
    p.add_argument("--compare_no3di", action="store_true", help="Also score a no3Di baseline (all 3Di set to tag).")
    p.add_argument("--tag", default="#", help="3Di tag for no3Di baseline (single char).")

    p.add_argument("--pdb_path", default=None, help="Optional PDB path used to extract angles / 3Di.")
    p.add_argument("--pdb_chain", default="A", help="Chain ID for structure extraction.")
    p.add_argument("--foldseek_path", default=None, help="Foldseek executable path (required for --use_foldseek_sequences).")
    p.add_argument("--plddt_mask", default="auto", help="Mask low-confidence residues in foldseek 3Di: auto/true/false.")
    p.add_argument("--plddt_threshold", type=float, default=70.0)

    p.add_argument("--use_struct_embeddings", default="auto", help="auto/true/false: include structural angles in model input.")
    p.add_argument("--num_struct_embeddings_layers", type=int, default=1)
    p.add_argument("--mask_angle_inputs_with_plddt", action="store_true", help="Mask angle inputs using pLDDT <= 70.")
    p.add_argument("--add_foldseek_embeddings", action="store_true", help="Add separate foldseek embeddings (advanced; rarely needed).")

    p.add_argument("--code_dir", default="/opt/ml/processing/input/code")
    p.add_argument("--data_dir", default="/opt/ml/processing/input/data")
    p.add_argument("--output_dir", default="/opt/ml/processing/output")
    p.add_argument("--pdb_dir", default="/opt/ml/processing/input/pdb")

    p.add_argument("--debug_alignment", action="store_true")
    p.add_argument("--debug_rows", type=int, default=1)
    p.add_argument("--debug_token_window", type=int, default=3)

    args = p.parse_args()

    if len(str(args.tag)) != 1:
        raise ValueError("--tag must be a single character")

    # Resolve I/O paths
    code_dir = Path(args.code_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = resolve_csv_paths(data_dir=data_dir, csv=args.input_csv)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    # Parse structural embedding flag
    s = str(args.use_struct_embeddings).strip().lower()
    if s == "auto":
        use_struct_embeddings = bool(args.pdb_path)
    elif s in {"1", "true", "t", "yes", "y"}:
        use_struct_embeddings = True
    elif s in {"0", "false", "f", "no", "n"}:
        use_struct_embeddings = False
    else:
        raise ValueError("--use_struct_embeddings must be auto/true/false")

    if args.use_foldseek_sequences and not args.pdb_path:
        raise ValueError("--use_foldseek_sequences requires --pdb_path")

    print_runtime_environment()

    model, tokenizer, _fs_tokenizer, device = load_model_and_tokenizers(
        model_id_or_path=args.model,
        use_struct_embeddings=use_struct_embeddings,
        num_struct_embeddings_layers=int(args.num_struct_embeddings_layers),
        mask_angle_inputs_with_plddt=bool(args.mask_angle_inputs_with_plddt),
        use_foldseek_sequences=bool(args.use_foldseek_sequences),
        add_foldseek_embeddings=bool(args.add_foldseek_embeddings),
        esm_checkpoint=args.esm_checkpoint,
        foldseek_base=str(args.foldseek_base),
    )

    pad_value = float(StructureTokenizer(use_foldseek_sequences=False).pad_value)

    pdb_wt_seq = None
    angles_for_model = None
    plddts = None
    foldseek_combined_seq = None

    if args.pdb_path:
        pdb_path = Path(args.pdb_path)
        if not pdb_path.exists():
            pdb_path = next(
                (p for p in (Path(args.pdb_dir) / args.pdb_path, data_dir / args.pdb_path, code_dir / args.pdb_path) if p.exists()),
                pdb_path,
            )
        if not pdb_path.exists():
            raise FileNotFoundError(f"pdb_path not found: {args.pdb_path!r}")

        pdb_wt_seq, angles_for_model, plddts, _coords = build_pdb_struct_context(
            pdb_path=pdb_path, pdb_chain=str(args.pdb_chain), pad_value=pad_value
        )

        if args.use_foldseek_sequences:
            foldseek_path = None
            if args.foldseek_path:
                foldseek_path = Path(args.foldseek_path)
                if not foldseek_path.exists():
                    foldseek_path = code_dir / "bin" / args.foldseek_path
            else:
                foldseek_path = code_dir / "bin" / "foldseek"
            if not foldseek_path.exists():
                raise FileNotFoundError(f"foldseek not found: {foldseek_path} (pass --foldseek_path)")

            s = "auto" if args.plddt_mask is None else str(args.plddt_mask).strip().lower()
            if s == "auto":
                plddt_mask = "auto"
            elif s in {"1", "true", "t", "yes", "y"}:
                plddt_mask = True
            elif s in {"0", "false", "f", "no", "n"}:
                plddt_mask = False
            else:
                raise ValueError("--plddt_mask must be auto/true/false")

            foldseek_combined_seq = build_foldseek_combined_seq(
                foldseek_path=foldseek_path,
                pdb_path=pdb_path,
                pdb_chain=str(args.pdb_chain),
                plddt_mask=plddt_mask,
                plddt_threshold=float(args.plddt_threshold),
            )

    summaries: list[dict] = []
    for csv_path in csv_paths:
        rec = run_one_csv(
            csv_path=csv_path,
            output_dir=output_dir,
            output_suffix=str(args.output_suffix),
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=int(args.batch_size),
            use_foldseek_sequences=bool(args.use_foldseek_sequences),
            use_struct_embeddings=bool(use_struct_embeddings),
            compare_no3di=bool(args.compare_no3di),
            tag=str(args.tag),
            pdb_wt_seq=pdb_wt_seq,
            foldseek_combined_seq=foldseek_combined_seq,
            angles_for_model=angles_for_model,
            plddts=plddts,
            pad_value=pad_value,
            progress_every=int(args.progress_every),
            debug_alignment=bool(args.debug_alignment),
            debug_rows=max(1, int(args.debug_rows)),
            debug_token_window=max(1, int(args.debug_token_window)),
        )
        if rec is not None:
            summaries.append(rec)

    if summaries:
        summary_path = output_dir / "summary.csv"
        pd.DataFrame(summaries).to_csv(summary_path, index=False)
        print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
