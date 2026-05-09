"""
online_ts.py — Online Thompson Sampling with localcolabfold on Frontera SLURM.

Each round:
  1. Select batch_size sequences via cluster-aware Thompson sampling.
  2. Split into FASTA batch files of seqs_per_job sequences each.
  3. Generate and optionally submit a SLURM job per batch file.
  4. Wait for all batch jobs to produce PDB outputs.
  5. Score outputs; update cluster posteriors.
  6. Checkpoint state for robust restart.

Run with --resume to continue from an existing checkpoint.
Run with --dry_run to prepare FASTA and SLURM files without submitting jobs.
"""

import argparse
import hashlib
import json
import logging
import math
import re
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from TS import (
    build_cluster_index,
    ensure_parent,
    initialize_rng,
    label_from_passes,
    load_config,
    save_outputs,
    select_batch,
)

LOG = logging.getLogger(__name__)
CHECKPOINT_VERSION = 1

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Online Thompson Sampling: select peptides each round, run "
            "localcolabfold via Frontera SLURM, score PDB outputs, and update "
            "cluster posteriors. Supports checkpointing and restart."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=Path, required=True,
        help="JSON config (TS params + optional 'slurm' section).",
    )
    p.add_argument(
        "--library", type=Path, required=True,
        help="CSV with columns: id, fasta_sequence, cluster.",
    )
    p.add_argument(
        "--work_root", type=Path, default=Path("runs/online_ts"),
        help="Root directory for round subdirectories.",
    )
    p.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Checkpoint JSON path. Default: <work_root>/checkpoint.json.",
    )
    p.add_argument(
        "--data_out", type=Path, default=Path("data.csv"),
        help="Aggregated label table (name, cluster, label).",
    )
    p.add_argument(
        "--metrics_out", type=Path, default=Path("af2_metrics.csv"),
        help="Detailed per-query ColabFold metrics.",
    )
    p.add_argument(
        "--out_prefix", type=Path, default=Path("online_ts"),
        help="Prefix for TS selections, curve, summary, and plot files.",
    )
    p.add_argument(
        "--seqs_per_job", type=int, default=None,
        help="Sequences per SLURM job (overrides config). Default: 10.",
    )
    p.add_argument(
        "--max_concurrent_jobs", type=int, default=None,
        help="Max concurrent SLURM jobs (overrides config). Default: 4.",
    )
    p.add_argument(
        "--batch_size", type=int, default=None,
        help="Sequences per round (overrides config.batch_size). Default: 50.",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Prepare FASTA and SLURM files without submitting jobs.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume from an existing checkpoint file.",
    )
    p.add_argument(
        "--submit_cmd", type=str, default="sbatch",
        help="SLURM submission command.",
    )
    p.add_argument(
        "--msa_template", type=Path, default=None,
        help=(
            "Precomputed protein MSA A3M file (e.g., no_pep.a3m). "
            "When provided, one A3M per peptide is written in a batch directory "
            "and ColabFold is invoked on that directory instead of a FASTA file. "
            "Can also be set via the 'msa_template' key in the JSON config."
        ),
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Library
# ─────────────────────────────────────────────────────────────────────────────


def load_library(path):
    df = pd.read_csv(path).reset_index(drop=True)
    required = ["id", "fasta_sequence", "cluster"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Library missing columns: {', '.join(missing)}")
    if df["id"].duplicated().any():
        dupes = df.loc[df["id"].duplicated(), "id"].tolist()[:5]
        raise ValueError(f"Library has duplicate ids: {dupes}")
    return df


def file_md5(path):
    h = hashlib.md5()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────────────────────────────────────


def _clean_for_json(obj):
    """Replace float NaN/inf with None so the output is valid JSON."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────


def empty_state(library_path, library_hash, rng_seed, alpha0, beta0, cluster_to_idx):
    return {
        "version": CHECKPOINT_VERSION,
        "library_path": str(library_path),
        "library_hash": library_hash,
        "rng_seed": rng_seed,
        "rng_state": None,
        "seed_done": False,
        "current_round": 0,
        "seen_ids": [],
        "alpha": {c: float(alpha0) for c in cluster_to_idx},
        "beta": {c: float(beta0) for c in cluster_to_idx},
        "queried_count": 0,
        "binder_count": 0,
        "curve_rows": [[0, 0]],
        "result_rows": [],
        "metric_rows": [],
        "rounds": [],
    }


def load_checkpoint(path):
    p = Path(path)
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def save_checkpoint(path, state):
    ensure_parent(path)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", newline="\n") as f:
        json.dump(_clean_for_json(state), f, indent=2)
    tmp.replace(Path(path))
    LOG.debug("Checkpoint saved → %s", path)


def _serialize_rng_state(state_dict):
    d = dict(state_dict)
    if "state" in d and isinstance(d["state"], dict):
        d["state"] = dict(d["state"])
    return d


def restore_rng(state):
    rng = np.random.default_rng(state["rng_seed"])
    if state.get("rng_state") is not None:
        rng.bit_generator.state = state["rng_state"]
    return rng


def capture_rng_state(rng):
    return _serialize_rng_state(rng.bit_generator.state)


# ─────────────────────────────────────────────────────────────────────────────
# Sequence I/O
# ─────────────────────────────────────────────────────────────────────────────


def sanitize_id(value):
    """Make a sequence ID safe for filenames, consistent with ColabFold naming."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "query"


def write_fasta(sequences, path):
    """Write list of {id, fasta_sequence} dicts as a FASTA file."""
    with Path(path).open("w", newline="\n") as f:
        for seq in sequences:
            f.write(f">{sanitize_id(seq['id'])}\n{seq['fasta_sequence']}\n")


def load_protein_msa(a3m_path, peptide_chain="A"):
    """
    Extract the protein MSA from a ColabFold complex A3M (e.g., no_pep.a3m).

    Reads the chain B unpaired block, strips the leading peptide-column
    gap prefix from every row, and returns the bare protein sequences so
    they can be re-padded for peptides of any length.

    Returns
    -------
    prot_query : str
        Pure protein query sequence (no gap characters).
    prot_msa_rows : list of (header_str, aligned_seq_str)
        Additional protein MSA rows with the pep-column prefix stripped.
    """
    lines = Path(a3m_path).read_text(encoding="utf-8").splitlines()
    comment = next((l for l in lines if l.startswith("#")), "")
    m = re.match(r"#\s*(\d+)\s*,\s*(\d+)", comment)
    if not m:
        raise ValueError(f"Cannot parse chain lengths from {a3m_path!r}: {comment!r}")
    len_a, len_b = int(m.group(1)), int(m.group(2))
    pep_len = len_a if peptide_chain.upper() == "A" else len_b

    # Locate chain B sentinel: ColabFold uses >102 (or >2 in older versions).
    # Match any ">N" where N % 100 == 2, which covers both >2 and >102.
    _num_hdr = re.compile(r"^>(\d+)$")

    def _is_chain_b(ln):
        m2 = _num_hdr.match(ln.strip())
        return m2 is not None and int(m2.group(1)) % 100 == 2

    idx_b = next((i for i, ln in enumerate(lines) if _is_chain_b(ln)), None)
    if idx_b is None:
        raise ValueError(f"Chain B sentinel (>102 or >2) not found in {a3m_path!r}")

    entries, hdr, parts = [], None, []
    for ln in lines[idx_b:]:
        if ln.startswith(">"):
            if hdr is not None:
                entries.append((hdr, "".join(parts)))
            hdr, parts = ln, []
        elif ln.strip():
            parts.append(ln.strip())
    if hdr is not None:
        entries.append((hdr, "".join(parts)))

    if not entries:
        raise ValueError(f"No chain B sequences found in {a3m_path!r}")

    # First entry is the chain B query; strip the pep_len leading dashes
    prot_query = entries[0][1][pep_len:]
    prot_msa_rows = [(h, s[pep_len:]) for h, s in entries[1:]]
    return prot_query, prot_msa_rows


def write_a3m_for_peptide(pep_seq, prot_query, prot_msa_rows, output_path):
    """
    Write a ColabFold multimer A3M for one peptide using a precomputed protein MSA.

    Layout
    ------
    # {L_pep},{L_prot}\\t1,1
    >101\\t102          <- paired block (query only)
    {pep_seq}{prot_query}
    >101               <- chain A (peptide) unpaired
    {pep_seq}{prot_gaps}
    >102               <- chain B (protein) unpaired
    {pep_gaps}{prot_query}
    ... protein MSA rows prepended with pep_gaps ...
    """
    L_pep = len(pep_seq)
    L_prot = len(prot_query)
    pep_gaps = "-" * L_pep
    prot_gaps = "-" * L_prot
    out = [
        f"#{L_pep},{L_prot}\t1,1",
        ">101\t102",
        f"{pep_seq}{prot_query}",
        ">101",
        f"{pep_seq}{prot_gaps}",
        ">102",
        f"{pep_gaps}{prot_query}",
    ]
    for hdr, seq in prot_msa_rows:
        out.append(hdr)
        out.append(f"{pep_gaps}{seq}")
    Path(output_path).write_text("\n".join(out) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# SLURM script generation
# ─────────────────────────────────────────────────────────────────────────────

_SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -o out.%j.out
#SBATCH -e err.%j.err
#SBATCH -t {walltime}
#SBATCH -N {nodes}
#SBATCH -n {ntasks}
#SBATCH -p {partition}
#SBATCH -A {account}
{mail_lines}
{module_setup}

export XLA_PYTHON_CLIENT_PREALLOCATE=false

{colabfold_cmd} \\
    {extra_args}{input_path} \\
    {output_dir}
"""


def generate_slurm_script(input_path, output_dir, job_name, slurm_cfg):
    """
    Return SLURM script text for a localcolabfold batch job (Frontera style).

    input_path may be a FASTA file (ColabFold generates MSAs) or a directory
    of A3M files (ColabFold uses precomputed MSAs).
    """
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    extra = slurm_cfg.get("extra_colabfold_args", "").strip()
    extra_arg_line = f"{extra} \\\n    " if extra else ""
    module_setup = slurm_cfg.get("module_setup", "").strip() or "# No module setup configured."
    mail_type = slurm_cfg.get("mail_type", "").strip()
    mail_user = slurm_cfg.get("mail_user", "").strip()
    if mail_type and mail_user:
        mail_lines = f"#SBATCH --mail-type={mail_type}\n#SBATCH --mail-user={mail_user}"
    else:
        mail_lines = ""
    return _SLURM_TEMPLATE.format(
        job_name=job_name,
        walltime=slurm_cfg.get("walltime", "02:00:00"),
        nodes=int(slurm_cfg.get("nodes", 1)),
        ntasks=int(slurm_cfg.get("ntasks", 4)),
        partition=slurm_cfg.get("partition", "rtx"),
        account=slurm_cfg.get("account", "YOUR_ALLOCATION"),
        mail_lines=mail_lines,
        module_setup=module_setup,
        colabfold_cmd=slurm_cfg.get("colabfold_cmd", "colabfold_batch"),
        extra_args=extra_arg_line,
        input_path=input_path,
        output_dir=Path(output_dir),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Job submission and tracking
# ─────────────────────────────────────────────────────────────────────────────


def submit_job(slurm_path, submit_cmd, dry_run):
    """Submit a SLURM script. Returns job_id string or None on dry_run."""
    if dry_run:
        LOG.info("[DRY RUN] Would submit: %s", slurm_path)
        return None
    cmd = submit_cmd.split() + [str(slurm_path.name)]
    result = subprocess.run(
        cmd, cwd=slurm_path.parent, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch failed for {slurm_path}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    m = re.search(r"Submitted batch job (\d+)", result.stdout)
    job_id = m.group(1) if m else None
    LOG.info("Submitted %s → job_id=%s", slurm_path.name, job_id)
    return job_id


def active_job_count(job_ids):
    """Count how many job_ids are still running/pending according to squeue."""
    live_ids = [j for j in job_ids if j is not None]
    if not live_ids:
        return 0
    result = subprocess.run(
        ["squeue", "-h", "-j", ",".join(live_ids), "-o", "%i"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return 0
    found = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return len(found & set(live_ids))


def wait_for_capacity(tracked_ids, max_concurrent, dry_run, poll_seconds):
    """Block until fewer than max_concurrent jobs are active."""
    if dry_run or max_concurrent <= 0:
        return
    while True:
        count = active_job_count(tracked_ids)
        if count < max_concurrent:
            return
        LOG.info(
            "Active jobs: %d/%d max — waiting %ds...", count, max_concurrent, poll_seconds
        )
        time.sleep(poll_seconds)


def _outputs_exist(output_dir, pdb_glob):
    odir = Path(output_dir)
    return bool(list(odir.glob(pdb_glob)) or list(odir.glob("**/*.pdb")))


def wait_for_round(batches, pdb_glob, poll_seconds, timeout_hours, dry_run):
    """Poll until every submitted batch output_dir has PDB files, or timeout."""
    if dry_run:
        return
    pending = {
        b["output_dir"]: b["job_id"]
        for b in batches
        if b["status"] == "submitted"
    }
    deadline = time.time() + timeout_hours * 3600.0
    while pending:
        done = [d for d in pending if _outputs_exist(d, pdb_glob)]
        for d in done:
            del pending[d]
        if not pending:
            break
        if time.time() >= deadline:
            LOG.warning("Timeout: %d batch output dirs still pending", len(pending))
            break
        active = active_job_count(list(pending.values()))
        LOG.info("Waiting: %d dirs pending, %d jobs active...", len(pending), active)
        time.sleep(poll_seconds)


# ─────────────────────────────────────────────────────────────────────────────
# PDB scoring (self-contained; no import from online.py)
# ─────────────────────────────────────────────────────────────────────────────


def _chain_residues(structure, chain_id):
    res = []
    for model in structure:
        if chain_id in model:
            res.extend(list(model[chain_id]))
    return res


def _ca_plddts(residues):
    return [float(r["CA"].get_bfactor()) for r in residues if "CA" in r]


def _longest_run_above(values, threshold):
    best, cur = [], []
    for i, v in enumerate(values):
        if v > threshold:
            cur.append(i)
            if len(cur) > len(best):
                best = cur[:]
        else:
            cur = []
    return best


def _rog(residues):
    coords = [a.get_coord() for r in residues for a in r]
    if not coords:
        return float("nan")
    c = np.asarray(coords, dtype=float)
    return float(np.sqrt(np.mean(np.sum((c - c.mean(0)) ** 2, axis=1))))


def score_single_pdb(pdb_path, cfg_ns, key_residues):
    """
    Score one PDB file.

    Returns dict with pep_plddt, avg_dist, rg_peptide, pass, status.
    If key_residues is empty, scoring is pLDDT-only (no distance check).
    cfg_ns must expose: peptide_chain, target_chain, plddt_threshold,
    dist_threshold, plddt_run_length.
    """
    from Bio.PDB import PDBParser

    structure = PDBParser(QUIET=True).get_structure("p", str(pdb_path))
    pep_res = _chain_residues(structure, cfg_ns.peptide_chain)
    if not pep_res:
        return {
            "pep_plddt": float("nan"), "avg_dist": float("nan"),
            "rg_peptide": float("nan"), "pass": 0,
            "status": f"NO_CHAIN_{cfg_ns.peptide_chain}",
        }
    plddts = _ca_plddts(pep_res)
    if not plddts:
        return {
            "pep_plddt": float("nan"), "avg_dist": float("nan"),
            "rg_peptide": float("nan"), "pass": 0, "status": "NO_CA",
        }
    rg = _rog(pep_res)
    run_idx = _longest_run_above(plddts, cfg_ns.plddt_threshold)
    if len(run_idx) >= cfg_ns.plddt_run_length:
        pep_plddt = float(np.mean([plddts[i] for i in run_idx]))
        scoring_res = [pep_res[i] for i in run_idx if i < len(pep_res)]
    else:
        pep_plddt = float(np.mean(plddts))
        scoring_res = pep_res

    avg_dist = float("nan")
    if key_residues:
        tgt_res = _chain_residues(structure, cfg_ns.target_chain)
        tgt_by_seqnum = {r.get_id()[1]: r for r in tgt_res}
        dists = []
        for _, res_idx in key_residues:
            tgt = tgt_by_seqnum.get(res_idx)
            if tgt is None or "CA" not in tgt:
                continue
            tgt_ca = tgt["CA"]
            for pr in scoring_res:
                if "CA" in pr:
                    dists.append(float(tgt_ca - pr["CA"]))
        avg_dist = float(np.mean(dists)) if dists else float("nan")

    if key_residues:
        passed = int(
            pep_plddt >= cfg_ns.plddt_threshold
            and not math.isnan(avg_dist)
            and avg_dist < cfg_ns.dist_threshold
        )
    else:
        passed = int(pep_plddt >= cfg_ns.plddt_threshold)

    return {
        "pep_plddt": pep_plddt, "avg_dist": avg_dist,
        "rg_peptide": rg, "pass": passed, "status": "OK",
    }


def find_sequence_pdbs(output_dir, safe_id, models_per_query):
    """
    Find PDB files for one sequence in a ColabFold output directory.

    ColabFold names files: {safe_id}_rank_001_alphafold2_*.pdb
    Returns a list of paths sorted by rank, up to models_per_query.
    """
    odir = Path(output_dir)
    pdbs = list(odir.glob(f"{safe_id}_rank_*.pdb"))
    if not pdbs:
        pdbs = list(odir.glob(f"{safe_id}*.pdb"))

    def _rank_key(p):
        m = re.search(r"rank[_-]?0*(\d+)", p.name)
        return int(m.group(1)) if m else 999

    return sorted(pdbs, key=_rank_key)[:models_per_query]


def score_sequence_pdbs(seq_id, cluster, pdb_files, cfg_ns, key_residues):
    """Score up to models_per_query PDB files for one sequence. Returns metric record."""
    record = {"id": seq_id, "cluster": cluster}
    for i in range(cfg_ns.models_per_query):
        record.update({
            f"pdb_{i}": "", f"plddt_{i}": float("nan"), f"dist_{i}": float("nan"),
            f"rg_{i}": float("nan"), f"pass_{i}": 0, f"status_{i}": "MISSING",
        })
    model_passes = []
    for i, pdb_file in enumerate(pdb_files[: cfg_ns.models_per_query]):
        record[f"pdb_{i}"] = str(pdb_file)
        try:
            s = score_single_pdb(pdb_file, cfg_ns, key_residues)
        except Exception as exc:
            s = {
                "pep_plddt": float("nan"), "avg_dist": float("nan"),
                "rg_peptide": float("nan"), "pass": 0, "status": f"ERROR:{exc}",
            }
        record[f"plddt_{i}"] = s["pep_plddt"]
        record[f"dist_{i}"] = s["avg_dist"]
        record[f"rg_{i}"] = s["rg_peptide"]
        record[f"pass_{i}"] = int(s["pass"])
        record[f"status_{i}"] = s["status"]
        model_passes.append(int(s["pass"]))
    record["label"] = label_from_passes(model_passes, cfg_ns.binder_rule)
    record["models_found"] = len(pdb_files)
    return record


# ─────────────────────────────────────────────────────────────────────────────
# Key-residue parsing
# ─────────────────────────────────────────────────────────────────────────────


def parse_key_residues(raw):
    """
    Parse 'I42,E43,I44' into [(label, int_index), ...].

    Returns [] for empty or missing input, enabling pLDDT-only scoring.
    """
    if not raw:
        return []
    raw = str(raw).strip()
    if not raw:
        return []
    residues = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        m = re.search(r"(\d+)$", token)
        if not m:
            raise ValueError(f"Could not parse residue index from '{token}'")
        residues.append((token, int(m.group(1))))
    return residues


# ─────────────────────────────────────────────────────────────────────────────
# Round preparation
# ─────────────────────────────────────────────────────────────────────────────


def prepare_batches(
    round_dir, batch_rows, seqs_per_job, round_index, slurm_cfg,
    prot_query=None, prot_msa_rows=None,
):
    """
    Write input files and SLURM scripts for one round.

    FASTA mode (default): writes batch_NNN.fasta; ColabFold generates MSAs.
    A3M mode (prot_query provided): writes batch_NNN_msas/ directory with one
    {safe_id}.a3m per peptide; ColabFold uses the precomputed protein MSA.

    Returns list of batch dicts with keys: fasta (FASTA mode) or msa_dir
    (A3M mode), slurm, output_dir, seq_ids, safe_ids, clusters, job_id, status.
    """
    use_a3m = prot_query is not None
    rows = list(batch_rows.iterrows())
    n_batches = math.ceil(len(rows) / seqs_per_job)
    batches = []
    for bi in range(n_batches):
        chunk = rows[bi * seqs_per_job: (bi + 1) * seqs_per_job]
        num = bi + 1
        label = "seed" if round_index < 0 else f"r{round_index:03d}"
        job_name = f"ts_{label}_b{num:03d}"

        slurm_path = round_dir / f"batch_{num:03d}.slurm"
        output_dir = round_dir / f"batch_{num:03d}_output"
        output_dir.mkdir(exist_ok=True)

        seq_ids = [str(row["id"]) for _, row in chunk]
        safe_ids = [sanitize_id(sid) for sid in seq_ids]
        clusters_map = {str(row["id"]): str(row["cluster"]) for _, row in chunk}

        if use_a3m:
            msa_dir = round_dir / f"batch_{num:03d}_msas"
            msa_dir.mkdir(exist_ok=True)
            for safe_id, (_, row) in zip(safe_ids, chunk):
                write_a3m_for_peptide(
                    str(row["fasta_sequence"]),
                    prot_query,
                    prot_msa_rows,
                    msa_dir / f"{safe_id}.a3m",
                )
            input_path = msa_dir
            batch_entry = {"msa_dir": str(msa_dir), "fasta": None}
            LOG.info(
                "Prepared batch %d/%d (A3M): %d seqs → %s/",
                num, n_batches, len(chunk), msa_dir.name,
            )
        else:
            fasta_path = round_dir / f"batch_{num:03d}.fasta"
            write_fasta(
                [{"id": safe_ids[i], "fasta_sequence": str(row["fasta_sequence"])}
                 for i, (_, row) in enumerate(chunk)],
                fasta_path,
            )
            input_path = fasta_path
            batch_entry = {"fasta": str(fasta_path), "msa_dir": None}
            LOG.info(
                "Prepared batch %d/%d: %d seqs → %s",
                num, n_batches, len(chunk), fasta_path.name,
            )

        slurm_path.write_text(
            generate_slurm_script(input_path, output_dir, job_name, slurm_cfg),
            encoding="utf-8",
        )
        batches.append({
            **batch_entry,
            "slurm": str(slurm_path),
            "output_dir": str(output_dir),
            "seq_ids": seq_ids,
            "safe_ids": safe_ids,
            "clusters": clusters_map,
            "job_id": None,
            "status": "pending",
        })
    return batches


def submit_batches(batches, tracked_ids, submit_cmd, max_concurrent, dry_run, poll_seconds):
    """Submit SLURM jobs for all pending batches, honouring the concurrency limit."""
    for batch in batches:
        if batch["status"] != "pending":
            continue
        wait_for_capacity(tracked_ids, max_concurrent, dry_run, poll_seconds)
        try:
            jid = submit_job(Path(batch["slurm"]), submit_cmd, dry_run)
            batch["job_id"] = jid
            batch["status"] = "submitted"
            if jid:
                tracked_ids.append(jid)
        except RuntimeError as exc:
            LOG.error("Submit failed for %s: %s", batch["slurm"], exc)
            batch["status"] = "submit_failed"
    return batches


def collect_round_results(batches, cfg_ns, key_residues, dry_run):
    """
    Score ColabFold outputs for all sequences in a round.

    Returns (metric_records, failed_ids).
    """
    if dry_run:
        return [], []
    metric_records, failed_ids = [], []
    for batch in batches:
        odir = batch["output_dir"]
        for seq_id, safe_id in zip(batch["seq_ids"], batch["safe_ids"]):
            cluster = batch["clusters"].get(seq_id, "unknown")
            pdbs = find_sequence_pdbs(odir, safe_id, cfg_ns.models_per_query)
            if not pdbs:
                LOG.warning("No PDB output for %s in %s", seq_id, odir)
                failed_ids.append(seq_id)
                continue
            try:
                rec = score_sequence_pdbs(seq_id, cluster, pdbs, cfg_ns, key_residues)
                metric_records.append(rec)
            except Exception as exc:
                LOG.error("Scoring error for %s: %s", seq_id, exc)
                failed_ids.append(seq_id)
    return metric_records, failed_ids


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    config = load_config(
        args.config,
        defaults={
            "alloc": "proportional",
            "random_state": 0,
            "models_per_query": 5,
            "pdb_glob": "*.pdb",
            "poll_seconds": 60,
            "job_timeout_hours": 72.0,
            "plddt_threshold": 70.0,
            "dist_threshold": 20.0,
            "plddt_run_length": 6,
            "key_residues": "",
            "peptide_chain": "A",
            "target_chain": "B",
            "binder_rule": "any",
            "seqs_per_job": 10,
            "max_concurrent_jobs": 4,
            "msa_template": "",  # empty → FASTA mode; set to path for precomputed MSA
        },
        allowed_binder_rules={"any", "top1", "majority"},
    )

    if args.seqs_per_job is not None:
        config["seqs_per_job"] = args.seqs_per_job
    if args.max_concurrent_jobs is not None:
        config["max_concurrent_jobs"] = args.max_concurrent_jobs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    cfg_ns = SimpleNamespace(**config)
    slurm_cfg = config.get("slurm", {})
    key_residues = parse_key_residues(cfg_ns.key_residues)

    # Resolve MSA template: CLI flag takes priority over config value.
    msa_template_path = args.msa_template or (
        Path(cfg_ns.msa_template) if cfg_ns.msa_template else None
    )
    prot_query = prot_msa_rows = None
    if msa_template_path:
        prot_query, prot_msa_rows = load_protein_msa(
            msa_template_path, cfg_ns.peptide_chain
        )
        LOG.info(
            "Loaded protein MSA from %s (%d rows, prot_len=%d)",
            msa_template_path, len(prot_msa_rows), len(prot_query),
        )

    args.work_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or (args.work_root / "checkpoint.json")

    library = load_library(args.library)
    lib_hash = file_md5(args.library)
    n = len(library)
    id_to_idx = {str(row["id"]): idx for idx, row in library.iterrows()}
    clusters_col, cluster_to_idx = build_cluster_index(library)

    # ── Load or create checkpoint ────────────────────────────────────────────
    state = None
    if args.resume:
        state = load_checkpoint(checkpoint_path)
        if state is None:
            LOG.warning(
                "--resume set but no checkpoint found at %s. Starting fresh.",
                checkpoint_path,
            )
        elif state.get("library_hash") != lib_hash:
            LOG.warning("Library hash mismatch (checkpoint vs current). Continuing anyway.")

    if state is None:
        rng_seed, rng = initialize_rng(cfg_ns.random_state)
        state = empty_state(
            args.library, lib_hash, rng_seed,
            cfg_ns.alpha0, cfg_ns.beta0, cluster_to_idx,
        )
    else:
        rng = restore_rng(state)

    # ── Restore TS state ─────────────────────────────────────────────────────
    seen = np.zeros(n, dtype=bool)
    seen_ids = set(state.get("seen_ids", []))
    for sid in seen_ids:
        if sid in id_to_idx:
            seen[id_to_idx[sid]] = True

    alpha = {c: float(v) for c, v in state.get("alpha", {}).items()}
    beta = {c: float(v) for c, v in state.get("beta", {}).items()}
    for c in cluster_to_idx:
        alpha.setdefault(c, float(cfg_ns.alpha0))
        beta.setdefault(c, float(cfg_ns.beta0))

    result_rows = state.get("result_rows", [])
    metric_rows = state.get("metric_rows", [])
    curve_rows = [tuple(r) for r in state.get("curve_rows", [[0, 0]])]
    queried_count = state.get("queried_count", 0)
    binder_count = state.get("binder_count", 0)

    tracked_ids = []  # job IDs submitted this session for concurrency tracking

    # ── Helpers ──────────────────────────────────────────────────────────────
    def absorb_records(metric_records, round_index):
        nonlocal queried_count, binder_count
        for rec in metric_records:
            seq_id = rec["id"]
            idx = id_to_idx.get(seq_id)
            if idx is None:
                LOG.warning("Scored ID %s not found in library; skipping.", seq_id)
                continue
            seen[idx] = True
            seen_ids.add(seq_id)
            cluster = clusters_col[idx]
            alpha[cluster] += int(rec["label"])
            beta[cluster] += 1 - int(rec["label"])
            queried_count += 1
            binder_count += int(rec["label"])
            curve_rows.append((queried_count, binder_count))
            result_rows.append({
                "round": round_index,
                "id": seq_id,
                "cluster": rec["cluster"],
                "label": int(rec["label"]),
            })
            metric_rows.append(rec)

    def flush_state():
        state["seen_ids"] = list(seen_ids)
        state["alpha"] = {k: float(v) for k, v in alpha.items()}
        state["beta"] = {k: float(v) for k, v in beta.items()}
        state["queried_count"] = queried_count
        state["binder_count"] = binder_count
        state["curve_rows"] = [list(r) for r in curve_rows]
        state["result_rows"] = result_rows
        state["metric_rows"] = metric_rows
        state["rng_state"] = capture_rng_state(rng)
        save_checkpoint(checkpoint_path, state)
        if result_rows:
            save_outputs(
                result_rows, metric_rows, list(curve_rows),
                args.data_out, args.metrics_out, args.out_prefix,
                state["rng_seed"], data_id_column="name",
            )

    # ── Resume any incomplete round from prior run ────────────────────────────
    for prior_round in state.get("rounds", []):
        if prior_round["status"] in ("complete", "dry_run"):
            continue
        if prior_round["status"] in ("submitted", "waiting"):
            round_index = prior_round["round"]
            LOG.info("Resuming incomplete round %d...", round_index)
            batches = prior_round.get("batches", [])
            wait_for_round(
                batches, cfg_ns.pdb_glob, cfg_ns.poll_seconds,
                cfg_ns.job_timeout_hours, args.dry_run,
            )
            metric_records, failed_ids = collect_round_results(
                batches, cfg_ns, key_residues, args.dry_run,
            )
            absorb_records(metric_records, round_index)
            for sid in failed_ids:
                if sid in id_to_idx:
                    seen[id_to_idx[sid]] = True
                    seen_ids.add(sid)
            prior_round["status"] = "complete"
            prior_round["scored_ids"] = [r["id"] for r in metric_records]
            prior_round["failed_ids"] = failed_ids
            flush_state()
            LOG.info(
                "[round %d] resumed — scored %d, failed %d",
                round_index, len(metric_records), len(failed_ids),
            )

    # ── Seed round ───────────────────────────────────────────────────────────
    if not state.get("seed_done", False):
        LOG.info("Running seed round (%d sequences)...", cfg_ns.seed_size)
        seed_order = np.arange(n)
        rng.shuffle(seed_order)
        seed_indices = [int(i) for i in seed_order if not seen[i]][: cfg_ns.seed_size]

        seed_df = library.loc[seed_indices].copy()
        round_dir = args.work_root / "round_seed"
        round_dir.mkdir(exist_ok=True)
        batches = prepare_batches(
            round_dir, seed_df, cfg_ns.seqs_per_job, -1, slurm_cfg,
            prot_query=prot_query, prot_msa_rows=prot_msa_rows,
        )

        if args.dry_run:
            LOG.info(
                "[DRY RUN] Seed: %d seqs in %d batches. Stopping here.",
                len(seed_indices), len(batches),
            )
            state["rng_state"] = capture_rng_state(rng)
            save_checkpoint(checkpoint_path, state)
            return

        seed_round_rec = {"round": -1, "status": "submitted", "batches": batches}
        state["rounds"].insert(0, seed_round_rec)
        batches = submit_batches(
            batches, tracked_ids, args.submit_cmd,
            cfg_ns.max_concurrent_jobs, args.dry_run, cfg_ns.poll_seconds,
        )
        state["rng_state"] = capture_rng_state(rng)
        save_checkpoint(checkpoint_path, state)

        wait_for_round(
            batches, cfg_ns.pdb_glob, cfg_ns.poll_seconds,
            cfg_ns.job_timeout_hours, args.dry_run,
        )
        metric_records, failed_ids = collect_round_results(
            batches, cfg_ns, key_residues, args.dry_run,
        )
        absorb_records(metric_records, -1)
        for sid in failed_ids:
            if sid in id_to_idx:
                seen[id_to_idx[sid]] = True
                seen_ids.add(sid)
        seed_round_rec["status"] = "complete"
        seed_round_rec["scored_ids"] = [r["id"] for r in metric_records]
        seed_round_rec["failed_ids"] = failed_ids
        state["seed_done"] = True
        flush_state()
        LOG.info(
            "[seed] done — scored %d, failed %d, binders so far %d",
            len(metric_records), len(failed_ids), binder_count,
        )

    # ── Main TS loop ─────────────────────────────────────────────────────────
    completed_rounds = {
        r["round"] for r in state.get("rounds", [])
        if r["status"] in ("complete", "dry_run")
    }
    start_round = state.get("current_round", 0)

    for round_index in range(start_round, cfg_ns.rounds):
        if round_index in completed_rounds:
            continue

        picked = select_batch(seen, cluster_to_idx, alpha, beta, cfg_ns, rng)
        if not picked:
            LOG.info("All clusters exhausted at round %d. Stopping.", round_index)
            break

        LOG.info("[round %d] selected %d sequences", round_index, len(picked))
        round_dir = args.work_root / f"round_{round_index:03d}"
        round_dir.mkdir(exist_ok=True)
        batch_df = library.loc[picked].copy()
        batches = prepare_batches(
            round_dir, batch_df, cfg_ns.seqs_per_job, round_index, slurm_cfg,
            prot_query=prot_query, prot_msa_rows=prot_msa_rows,
        )

        round_rec = {"round": round_index, "status": "pending", "batches": batches}
        state["rounds"].append(round_rec)
        state["current_round"] = round_index
        state["rng_state"] = capture_rng_state(rng)

        if args.dry_run:
            LOG.info(
                "[DRY RUN] Round %d: %d seqs in %d batches.",
                round_index, len(picked), len(batches),
            )
            round_rec["status"] = "dry_run"
            save_checkpoint(checkpoint_path, state)
            continue

        save_checkpoint(checkpoint_path, state)
        batches = submit_batches(
            batches, tracked_ids, args.submit_cmd,
            cfg_ns.max_concurrent_jobs, args.dry_run, cfg_ns.poll_seconds,
        )
        round_rec["status"] = "submitted"
        save_checkpoint(checkpoint_path, state)

        wait_for_round(
            batches, cfg_ns.pdb_glob, cfg_ns.poll_seconds,
            cfg_ns.job_timeout_hours, args.dry_run,
        )
        metric_records, failed_ids = collect_round_results(
            batches, cfg_ns, key_residues, args.dry_run,
        )
        absorb_records(metric_records, round_index)
        for sid in failed_ids:
            if sid in id_to_idx:
                seen[id_to_idx[sid]] = True
                seen_ids.add(sid)

        round_rec["status"] = "complete"
        round_rec["scored_ids"] = [r["id"] for r in metric_records]
        round_rec["failed_ids"] = failed_ids
        round_rec["batches"] = batches

        flush_state()
        LOG.info(
            "[round %d] queried=%d binders=%d rate=%.4f failed=%d",
            round_index, queried_count, binder_count,
            binder_count / max(queried_count, 1), len(failed_ids),
        )

    LOG.info("Finished. Wrote %s and %s.", args.data_out, args.metrics_out)


if __name__ == "__main__":
    main()
