# Dataset Generation Pipeline -- Implementation Plan

## Todo Item #10: Script to run Boltz1 on PDBs and extract pLDDT labels

---

## 1. Summary

Build `scripts/generate_dataset.py` that:
1. Reads PDB files from an input directory
2. Extracts protein sequences and chain IDs from each PDB
3. Generates Boltz-compatible YAML input files (single-sequence mode)
4. Runs `boltz predict --model boltz1` via subprocess
5. Parses pLDDT from output `.npz` files
6. Converts continuous pLDDT (0-1) to bin indices (0-49) for cross-entropy training
7. Saves per-PDB `.pt` label files to `data/processed/labels/`

---

## 2. Design Decisions

### 2.1 Subprocess vs Python API

**Decision: Subprocess call to `boltz predict` CLI.**

Rationale:
- The Boltz pip package (v2.2.1) exposes a Click CLI, not a clean Python API. The `predict` function signature is `(*args, **kwargs)` via Click decorators, making programmatic use fragile.
- Subprocess gives clear isolation -- no import conflicts between the vendored `src/boltz/` (confidence module subset) and the pip-installed `boltz` (full package).
- Subprocess is robust to Boltz version changes and easy to debug.
- Each PDB can be processed independently, enabling natural parallelism and resumability.

### 2.2 Single-Sequence vs MSA Server

**Decision: Single-sequence mode (`msa: empty`) by default, with `--use-msa-server` as an optional flag.**

Rationale:
- Training the adaptor needs pLDDT labels, not maximum-quality Boltz predictions. Single-sequence mode is faster and does not require network access.
- The Boltz YAML format supports `msa: empty` to explicitly request single-sequence mode.
- MSA-based predictions can be generated later if higher-quality labels are needed.
- The script accepts a `--use-msa-server` flag to enable MSA generation when desired.

### 2.3 PDB Parsing

**Decision: BioPython `PDBParser` + `Polypeptide`.**

Rationale:
- BioPython is already in the environment.
- `PDBParser` handles all PDB format quirks (multi-model, HETATM, insertion codes).
- `PPBuilder` or iterating over residues extracts standard amino acid sequences cleanly.
- Handles multi-chain PDBs naturally by iterating over chains.

### 2.4 Multi-Chain PDBs

**Decision: Include all protein chains in a single Boltz YAML input.**

Rationale:
- Boltz natively handles multi-chain inputs and produces per-token pLDDT for all chains.
- The YAML format supports multiple protein entries with different chain IDs.
- The pLDDT npz output contains per-token scores across all chains in input order.
- We store one `.pt` file per PDB (not per chain), with a chain mapping for downstream indexing.

### 2.5 What Metadata to Store

Each `{pdb_id}.pt` file contains:
```python
{
    "pdb_id": str,                    # PDB identifier (stem of input file)
    "sequences": dict[str, str],      # chain_id -> amino acid sequence
    "plddt": Tensor,                  # [N_total] float32, per-residue pLDDT (0-1 scale)
    "plddt_bin": Tensor,              # [N_total] int64, per-residue bin index (0-49)
    "chain_lengths": dict[str, int],  # chain_id -> number of residues
    "n_residues": int,                # total residues across all chains
}
```

The `plddt` field stores continuous values for analysis/visualization. The `plddt_bin` field stores bin indices for cross-entropy training loss.

---

## 3. pLDDT Binning Scheme

### How Boltz bins pLDDT (from `src/boltz/model/loss/confidence.py`)

The Boltz training loss computes bin indices from continuous lDDT values:
```python
num_bins = 50  # num_plddt_bins
bin_index = torch.floor(target_lddt * num_bins).long()
bin_index = torch.clamp(bin_index, max=(num_bins - 1))
```

This gives 50 uniform bins over [0, 1):
- Bin 0: lDDT in [0.00, 0.02)
- Bin 1: lDDT in [0.02, 0.04)
- ...
- Bin 49: lDDT in [0.98, 1.00]

The bin centers (used by `compute_aggregated_metric` for inference) are:
```python
bin_width = 1.0 / 50  # = 0.02
centers = [0.01, 0.03, 0.05, ..., 0.99]
```

### What Boltz outputs in the npz file

The pLDDT saved to `plddt_{name}_model_0.npz` is the **continuous aggregated metric** from `compute_aggregated_metric(plddt_logits)`, which is a weighted sum of bin centers by softmax probabilities. This gives values in approximately [0, 1].

### Our conversion

We apply the same binning as the Boltz training loss:
```python
def plddt_to_bin(plddt: Tensor, num_bins: int = 50) -> Tensor:
    """Convert continuous pLDDT (0-1) to bin indices (0 to num_bins-1)."""
    bin_index = torch.floor(plddt * num_bins).long()
    return torch.clamp(bin_index, min=0, max=num_bins - 1)
```

**Important note**: The Boltz output pLDDT is a *predicted* lDDT (from logits), not a *true* lDDT computed from coordinate comparison. For our training, we treat Boltz's predicted pLDDT as the ground-truth label for the adaptor to learn. This is by design -- we want the adaptor to reproduce Boltz's confidence predictions from La-Proteina features.

---

## 4. Boltz Input YAML Format

For each PDB, we generate a YAML file like:
```yaml
sequences:
  - protein:
      id: A
      sequence: "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPP..."
      msa: empty
  - protein:
      id: B
      sequence: "MADQLTEEQIAEFKEAFSLF..."
      msa: empty
version: 1
```

Key points:
- `msa: empty` forces single-sequence mode (no MSA search).
- Each chain gets its own protein entry.
- `version: 1` is required by the Boltz parser.
- Only protein chains are included (ligands, nucleic acids, water are skipped).

---

## 5. Script Design: `scripts/generate_dataset.py`

### CLI Interface

```
python scripts/generate_dataset.py \
    --input-dir data/raw/pdbs/ \
    --output-dir data/processed/labels/ \
    --work-dir data/processed/boltz_work/ \
    --model boltz1 \
    [--use-msa-server] \
    [--diffusion-samples 1] \
    [--sampling-steps 200] \
    [--recycling-steps 3] \
    [--devices 1] \
    [--accelerator gpu] \
    [--num-bins 50] \
    [--override] \
    [--single-pdb PATH]  # Process a single PDB file (for testing)
    [--wandb-project TEXT]  # W&B project name (default: "quality-graft")
    [--wandb-run-name TEXT] # W&B run name (default: auto-generated)
    [--wandb-entity TEXT]   # W&B entity/team
    [--no-wandb]            # Disable W&B logging
```

### Processing Pipeline (per PDB)

```
1. Parse PDB with BioPython
   |-- Extract chain IDs and sequences (standard amino acids only)
   |-- Skip non-protein chains (DNA, RNA, ligands, water)
   |-- Validate: at least 1 protein chain with >= 1 residue
   |
2. Generate Boltz YAML input
   |-- Write to work_dir/inputs/{pdb_id}.yaml
   |-- Include all protein chains with msa: empty
   |
3. Run Boltz prediction (subprocess)
   |-- boltz predict {yaml_path} --out_dir {work_dir}/boltz_out --model boltz1 ...
   |-- Capture stdout/stderr for logging
   |-- Check return code; raise on failure
   |
4. Parse Boltz output
   |-- Find plddt_{pdb_id}_model_0.npz in output directory
   |-- Load plddt array: np.load(path)["plddt"]
   |-- Verify shape matches total residue count
   |
5. Convert and save
   |-- Convert plddt to torch tensor
   |-- Compute bin indices: floor(plddt * 50), clamp to [0, 49]
   |-- Save .pt dict to output_dir/{pdb_id}.pt
   |
6. Log to W&B
   |-- Log per-protein scalar metrics (mean, median, percentiles, coverage fractions, segments)
   |-- Append to protein_stats accumulator for final summary
   |
7. Log success, move to next PDB

After all PDBs:
8. Generate and log W&B summary plots (distributions, relationship plots, positional heatmap, segment stats)
9. Log W&B summary table (one row per protein, interactive)
10. wandb.finish()
```

### Resumability

- Before processing each PDB, check if `output_dir/{pdb_id}.pt` already exists.
- If it exists and `--override` is not set, skip it.
- If `--override` is set, reprocess and overwrite.
- Failed PDBs are logged to `work_dir/failures.log` with error messages.

### Error Handling

| Error | Handling |
|---|---|
| PDB parse failure | Log warning, skip PDB, continue |
| No protein chains in PDB | Log warning, skip PDB, continue |
| Boltz subprocess failure (non-zero exit) | Log error with stderr, skip PDB, continue |
| Missing pLDDT npz in output | Log error, skip PDB, continue |
| Shape mismatch (pLDDT length != residue count) | Log warning with details, save anyway (Boltz may tokenize differently) |

---

## 6. File Structure

### New Files

```
scripts/
|-- generate_dataset.py          # Main dataset generation script

src/quality_graft/data/
|-- pdb_utils.py                 # PDB parsing utilities (extract sequences, chain info)
|-- boltz_runner.py              # Boltz subprocess runner + output parser
|-- plddt_utils.py               # pLDDT binning and label generation
|-- wandb_logger.py              # W&B init, per-protein metrics, summary plots

tests/
|-- test_dataset_generation.py   # Unit tests (PDB parsing, binning, label format)
|-- integration/
|   |-- test_generate_dataset.py # Heavy test: run on 1ubq.pdb (GPU + Boltz model)
```

### Output Structure

```
data/
|-- 1ubq.pdb                     # Test PDB (existing)
|-- processed/
|   |-- labels/                  # pLDDT labels
|   |   +-- 1ubq.pt             # {pdb_id, sequences, plddt, plddt_bin, ...}
|   |-- boltz_work/             # Working directory (can be deleted after)
|   |   |-- inputs/             # Generated YAML files
|   |   |   +-- 1ubq.yaml
|   |   |-- boltz_out/          # Raw Boltz output
|   |   |   +-- 1ubq/
|   |   |       +-- predictions/
|   |   |           +-- 1ubq/
|   |   |               |-- 1ubq_model_0.cif
|   |   |               |-- plddt_1ubq_model_0.npz
|   |   |               +-- confidence_1ubq_model_0.json
|   |   +-- failures.log
|   +-- splits/
|       |-- train.txt
|       |-- val.txt
|       +-- test.txt
```

---

## 7. Module Details

### 7.1 `src/quality_graft/data/pdb_utils.py`

```python
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ChainInfo:
    chain_id: str
    sequence: str
    n_residues: int

def parse_pdb_chains(pdb_path: Path) -> list[ChainInfo]:
    """Extract protein chain sequences from a PDB file.

    Uses BioPython PDBParser + standard residue filtering.
    Returns only protein chains (standard amino acids).
    Skips HETATM, water, non-standard residues.

    Raises ValueError if no protein chains found.
    """
    ...

def chains_to_boltz_yaml(chains: list[ChainInfo], use_msa: bool = False) -> str:
    """Generate Boltz-compatible YAML content from chain info.

    With use_msa=False, sets msa: empty for single-sequence mode.
    """
    ...
```

### 7.2 `src/quality_graft/data/boltz_runner.py`

```python
from pathlib import Path
from dataclasses import dataclass
import numpy as np

@dataclass
class BoltzResult:
    pdb_id: str
    plddt: np.ndarray         # [N_total] float, 0-1 scale
    confidence_json: dict     # Full confidence summary
    success: bool
    error_msg: str | None

def run_boltz_predict(
    yaml_path: Path,
    out_dir: Path,
    model: str = "boltz1",
    devices: int = 1,
    accelerator: str = "gpu",
    diffusion_samples: int = 1,
    sampling_steps: int = 200,
    recycling_steps: int = 3,
    use_msa_server: bool = False,
    override: bool = False,
) -> BoltzResult:
    """Run boltz predict as a subprocess and parse results.

    Returns BoltzResult with pLDDT array on success.
    On failure, returns BoltzResult with success=False and error_msg.
    """
    ...

def find_plddt_npz(boltz_out_dir: Path, pdb_id: str) -> Path | None:
    """Locate the pLDDT npz file in Boltz output directory.

    Expected at: boltz_out_dir/predictions/{pdb_id}/plddt_{pdb_id}_model_0.npz
    """
    ...
```

### 7.3 `src/quality_graft/data/plddt_utils.py`

```python
import torch
from torch import Tensor

NUM_PLDDT_BINS = 50

def plddt_to_bin(plddt: Tensor, num_bins: int = NUM_PLDDT_BINS) -> Tensor:
    """Convert continuous pLDDT values (0-1 scale) to bin indices.

    Uses the same binning as Boltz training loss:
        bin_index = floor(plddt * num_bins), clamped to [0, num_bins-1]

    Parameters
    ----------
    plddt : Tensor
        Continuous pLDDT values in [0, 1]. Any shape.
    num_bins : int
        Number of bins (default 50, matching Boltz config).

    Returns
    -------
    Tensor
        Integer bin indices, same shape as input. dtype=int64.
    """
    bin_index = torch.floor(plddt * num_bins).long()
    return torch.clamp(bin_index, min=0, max=num_bins - 1)

def bin_to_plddt(bin_index: Tensor, num_bins: int = NUM_PLDDT_BINS) -> Tensor:
    """Convert bin indices back to bin center values.

    Inverse of plddt_to_bin (approximate -- returns bin centers).
    Useful for evaluation/visualization.
    """
    bin_width = 1.0 / num_bins
    return (bin_index.float() + 0.5) * bin_width
```

---

## 8. Implementation Order

### Phase A: Core utilities (no GPU needed)

1. **`src/quality_graft/data/plddt_utils.py`** -- pLDDT binning functions
2. **`src/quality_graft/data/pdb_utils.py`** -- PDB parsing and YAML generation
3. **`tests/test_dataset_generation.py`** -- Unit tests for the above
   - Test `plddt_to_bin` with known values (0.0 -> bin 0, 0.5 -> bin 25, 1.0 -> bin 49)
   - Test `bin_to_plddt` round-trip
   - Test `parse_pdb_chains` on `data/1ubq.pdb` (76 residues, chain A)
   - Test `chains_to_boltz_yaml` format

### Phase B: Boltz runner (no GPU needed for code, but tests need GPU)

4. **`src/quality_graft/data/boltz_runner.py`** -- Subprocess runner + output parser
5. **`tests/test_dataset_generation.py`** additions -- Unit tests for path construction, CLI building

### Phase C: W&B logging module

6. **`src/quality_graft/data/wandb_logger.py`** -- W&B init, per-protein logging, summary plot generation
   - `init_wandb_run(args)` — creates run with config
   - `log_protein_metrics(pdb_id, plddt, n_residues, elapsed_s)` — per-step scalars
   - `log_dataset_summary(protein_stats)` — all matplotlib plots + wandb.Table
   - `finish_wandb_run()` — finalize
   - All functions are no-ops when `--no-wandb` is set (guard on `wandb.run is not None`)

### Phase D: Main script

7. **`scripts/generate_dataset.py`** -- Main CLI script wiring all modules together

### Phase E: Integration test (GPU + Boltz model required)

8. **`tests/integration/test_generate_dataset.py`** -- Heavy test on `1ubq.pdb`
   - Marked with `@pytest.mark.heavy`
   - Runs full pipeline on 1ubq.pdb
   - Verifies output `.pt` file exists with correct keys and shapes
   - Verifies pLDDT values are in [0, 1]
   - Verifies bin indices are in [0, 49]
   - Verifies residue count is 76 (ubiquitin)

---

## 9. Test Plan

### Unit Tests (`tests/test_dataset_generation.py`)

```python
class TestPlddtBinning:
    def test_boundary_values(self):
        # 0.0 -> bin 0, 0.5 -> bin 25, 0.99 -> bin 49, 1.0 -> bin 49 (clamped)

    def test_round_trip(self):
        # bin_to_plddt(plddt_to_bin(x)) should be close to x (within bin_width/2)

    def test_shape_preserved(self):
        # Input [B, N] -> output [B, N]

    def test_dtype(self):
        # Output should be int64

    def test_negative_clamped(self):
        # Negative values -> bin 0

class TestPdbParsing:
    def test_parse_1ubq(self):
        # Parse data/1ubq.pdb
        # Expect 1 chain (A), 76 residues
        # Sequence starts with MQIFVKTLTG...

    def test_yaml_generation(self):
        # Generate YAML from chain info
        # Verify YAML structure: sequences, protein, id, sequence, msa: empty, version: 1

    def test_yaml_single_sequence_mode(self):
        # Verify msa: empty appears in YAML output

class TestBoltzRunner:
    def test_command_construction(self):
        # Verify the subprocess command is built correctly
        # Check --model boltz1, --out_dir, etc.

    def test_npz_path_resolution(self):
        # Test find_plddt_npz path logic

class TestSegmentAnalysis:
    def test_longest_contiguous_below(self):
        # [0.8, 0.3, 0.2, 0.4, 0.9, 0.1] with threshold 0.5 -> longest = 3
        # All above threshold -> 0
        # All below threshold -> full length

    def test_count_segments_below(self):
        # [0.8, 0.3, 0.2, 0.4, 0.9, 0.1] with threshold 0.5 -> 2 segments
        # No segments -> 0

class TestWandbLogger:
    def test_no_wandb_mode(self):
        # With --no-wandb, all functions should be no-ops (no errors)

    def test_protein_metrics_computation(self):
        # Given a known pLDDT array, verify computed metrics match expected values
        # plddt = [0.95, 0.85, 0.45, 0.30, 0.72]
        # mean=0.654, f90=0.2, f70=0.4, f50=0.6, etc.
```

### Integration Test (`tests/integration/test_generate_dataset.py`)

```python
@pytest.mark.heavy
class TestGenerateDataset:
    def test_1ubq_end_to_end(self, tmp_path):
        """Run full pipeline on 1ubq.pdb and verify output."""
        # 1. Run generate_dataset.py on data/1ubq.pdb
        # 2. Check output file exists
        # 3. Load and verify contents:
        #    - pdb_id == "1ubq"
        #    - sequences["A"] starts with "MQIFVKTLTG"
        #    - plddt shape == [76]
        #    - plddt_bin shape == [76]
        #    - plddt values in [0, 1]
        #    - plddt_bin values in [0, 49]
        #    - n_residues == 76
        # 4. Verify W&B was called (use --no-wandb for test, or mock wandb)

    def test_1ubq_with_wandb(self, tmp_path):
        """Run pipeline on 1ubq.pdb with W&B in offline mode."""
        # Set WANDB_MODE=offline to avoid network calls
        # Verify wandb run dir is created with expected metrics
```

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Boltz tokenization differs from BioPython residue count | pLDDT length mismatch | Log warning, include both counts in .pt file. Boltz may add special tokens or handle modified residues differently. |
| Boltz model download on first run (~6.9 GB) | Slow first execution | Document in README. The model is cached in `~/.boltz/` after first download. |
| PYTHONPATH conflict between vendored `src/boltz/` and pip `boltz` | Import errors in subprocess | The subprocess runs in a clean shell -- no PYTHONPATH modification needed. The script itself only imports from `quality_graft.data.*`, not from `boltz`. |
| GPU memory for large proteins | OOM during Boltz inference | Boltz handles this internally with batching. For very large proteins (>1000 residues), `--devices` and `--max-parallel-samples` can be tuned. |
| PDB files with non-standard residues | Parse errors or missing residues | BioPython `is_aa(std=True)` filters to standard amino acids. Non-standard residues are skipped with a warning. |

---

## 11. Configuration Extension

The existing config stubs will be updated minimally:

### `configs/data/dataset.yaml` (extend, do not replace)
```yaml
# Dataset configuration
labels_dir: data/processed/labels/
splits_dir: data/processed/splits/
num_plddt_bins: 50
```

The preprocessing config remains a stub until task #11 (training data pipeline).

---

## 12. Quality Checklist

- [ ] No circular dependencies: `pdb_utils`, `boltz_runner`, `plddt_utils` are leaf modules with no cross-imports
- [ ] Clear ownership: PDB parsing in `pdb_utils`, Boltz execution in `boltz_runner`, binning in `plddt_utils`
- [ ] Testable: Each module testable independently; unit tests need no GPU; heavy tests clearly marked
- [ ] No hardcoded paths: All paths configurable via CLI args
- [ ] Error handling: Every failure mode has a defined behavior (skip + log)
- [ ] Resumable: Skip existing outputs by default
- [ ] PYTHONPATH isolation: Subprocess avoids vendored `src/boltz/` conflicts
- [ ] W&B logging: All metrics, distributions, and visualizations logged per-protein and at dataset level

---

## 13. Weights & Biases Integration

W&B tracks both system resources (CPU/GPU — automatic via `wandb.init()`) and custom dataset-generation metrics. The run is initialized once at script start; per-protein scalars are logged as each PDB completes; dataset-wide distributions and plots are logged as a final summary step.

### 13.1 Run Configuration

```python
import wandb

run = wandb.init(
    project="quality-graft",
    job_type="dataset-generation",
    config={
        "model": "boltz1",
        "diffusion_samples": args.diffusion_samples,
        "sampling_steps": args.sampling_steps,
        "recycling_steps": args.recycling_steps,
        "use_msa_server": args.use_msa_server,
        "num_plddt_bins": args.num_bins,
        "accelerator": args.accelerator,
        "input_dir": str(args.input_dir),
    },
)
```

W&B system metrics (CPU %, GPU %, GPU memory, etc.) are captured automatically.

### 13.2 Per-Protein Logging (streaming, as each PDB completes)

After each successful Boltz run, log scalar metrics to a W&B step:

```python
# plddt is a 1-D numpy array of per-residue pLDDT values (0-1 scale)
wandb.log({
    "protein/pdb_id": pdb_id,
    "protein/length": n_residues,
    "protein/mean_plddt": plddt.mean(),
    "protein/median_plddt": np.median(plddt),
    "protein/p10_plddt": np.percentile(plddt, 10),
    "protein/p25_plddt": np.percentile(plddt, 25),
    "protein/std_plddt": plddt.std(),
    "protein/iqr_plddt": np.percentile(plddt, 75) - np.percentile(plddt, 25),
    "protein/frac_ge90": (plddt >= 0.90).mean(),   # f90
    "protein/frac_ge70": (plddt >= 0.70).mean(),   # f70
    "protein/frac_lt50": (plddt < 0.50).mean(),     # f50
    "protein/L70": (plddt >= 0.70).sum(),            # high-confidence length
    "protein/L70_frac": (plddt >= 0.70).mean(),      # L70 / L
    "protein/longest_low_segment": longest_contiguous_below(plddt, 0.50),
    "protein/num_low_segments": count_segments_below(plddt, 0.50),
    "protein/nterm_30_mean": plddt[:30].mean() if n_residues >= 30 else plddt.mean(),
    "protein/cterm_30_mean": plddt[-30:].mean() if n_residues >= 30 else plddt.mean(),
    "protein/core_mean": plddt[30:-30].mean() if n_residues > 60 else plddt.mean(),
    "protein/boltz_walltime_s": elapsed_seconds,
    "progress/n_processed": n_processed,
    "progress/n_failed": n_failed,
    "progress/n_skipped": n_skipped,
})
```

### 13.3 Dataset-Wide Summary Plots (logged once after all PDBs)

All plots use `matplotlib` (already in the environment) and are logged as `wandb.Image`. Collected arrays: `all_means`, `all_medians`, `all_lengths`, `all_f90`, `all_f70`, `all_f50`, `all_stds`, `all_iqrs`, `all_residue_plddt` (pooled), etc.

#### Category 1: pLDDT Distributions

| Plot | Description |
|---|---|
| **Per-protein mean pLDDT histogram** | Histogram + violin of `all_means` with vertical lines at 0.50, 0.70, 0.90 cutoffs |
| **Per-protein median pLDDT histogram** | Same for medians |
| **Per-protein 10th/25th percentile histogram** | Distributions of p10 and p25 per protein |
| **Pooled per-residue pLDDT histogram** | Histogram of all residues' pLDDT values (dominated by long proteins) |
| **ECDF of per-protein mean pLDDT** | Cumulative plot answering "what fraction of proteins have mean pLDDT >= X?" with reference lines at 50, 70, 90 |

#### Category 2: Confidence Coverage

| Plot | Description |
|---|---|
| **f90 / f70 / f50 distributions** | Histogram/violin of per-protein fractions (3 subplots or overlaid) |
| **Stacked bar chart** | One bar per protein (sorted by mean pLDDT), 4 stacked segments: >=90, 70-90, 50-70, <50. For large datasets, subsample or use a heatmap-style representation. |
| **L70 / L distribution** | Histogram of `L70_frac` across proteins |

#### Category 3: Relationship Plots

| Plot | Description |
|---|---|
| **Length vs mean pLDDT** | Scatter/hexbin of protein length vs mean pLDDT |
| **Length vs f70** | Scatter/hexbin of protein length vs fraction >= 0.70 |
| **Length vs f50** | Scatter/hexbin of protein length vs fraction < 0.50 |
| **Mean vs std pLDDT** | Scatter of per-protein mean pLDDT vs std(pLDDT), colored by length |
| **Mean vs IQR pLDDT** | Scatter of per-protein mean pLDDT vs IQR(pLDDT) |

#### Category 4: Positional Summaries

| Plot | Description |
|---|---|
| **Average pLDDT vs relative position** | Bin residues by relative position (0-1, 100 bins) across all proteins, plot mean pLDDT per position bin |
| **N-term / Core / C-term comparison** | Box plot: first 30 aa, middle, last 30 aa mean pLDDT |
| **pLDDT heatmap** | Heatmap with proteins on y-axis (sorted by mean pLDDT), relative position on x-axis (100 bins), color = pLDDT. Each protein rescaled to 100 bins via interpolation. |

#### Category 5: Segment Statistics

| Plot | Description |
|---|---|
| **Longest low-confidence segment distribution** | Histogram of longest contiguous segment with pLDDT < 0.50 per protein |
| **Number of low-confidence segments distribution** | Histogram of count of contiguous segments with pLDDT < 0.50 per protein |

### 13.4 W&B Summary Table

Log a `wandb.Table` with one row per protein for interactive exploration in the W&B dashboard:

```python
table = wandb.Table(
    columns=[
        "pdb_id", "length", "mean_plddt", "median_plddt",
        "p10_plddt", "p25_plddt", "std_plddt", "iqr_plddt",
        "f90", "f70", "f50", "L70", "L70_frac",
        "longest_low_seg", "num_low_segs",
        "nterm_30_mean", "cterm_30_mean", "core_mean",
        "walltime_s",
    ],
    data=rows,  # list of lists, one per protein
)
wandb.log({"dataset/protein_table": table})
```

This enables scatter plots, filtering, and sorting directly in the W&B UI.

### 13.5 Helper Functions for Segment Analysis

```python
def longest_contiguous_below(plddt: np.ndarray, threshold: float) -> int:
    """Length of the longest contiguous run of residues with pLDDT < threshold."""
    below = plddt < threshold
    if not below.any():
        return 0
    # Encode run lengths
    changes = np.diff(below.astype(int), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return int((ends - starts).max())


def count_segments_below(plddt: np.ndarray, threshold: float) -> int:
    """Number of contiguous segments with pLDDT < threshold."""
    below = plddt < threshold
    if not below.any():
        return 0
    changes = np.diff(below.astype(int), prepend=0, append=0)
    return int((changes == 1).sum())
```

### 13.6 Module Placement

The W&B logging code lives in a dedicated module:

```
src/quality_graft/data/
|-- wandb_logger.py    # W&B init, per-protein logging, summary plots
```

This module exposes:
- `init_wandb_run(args) -> wandb.Run` — initialize the run with config
- `log_protein_metrics(pdb_id, plddt, n_residues, elapsed_s)` — per-protein step logging
- `log_dataset_summary(protein_stats: list[dict])` — generate and log all summary plots + table
- `finish_wandb_run()` — finalize and upload

The main script calls these at the appropriate points. The plotting functions use `matplotlib` and return `wandb.Image` objects.

### 13.7 W&B Dashboard Layout (recommended panels)

For the W&B project, create a workspace with these panel groups:

1. **Progress** — `progress/n_processed`, `progress/n_failed`, system CPU/GPU charts
2. **Per-Protein Live** — line charts of `protein/mean_plddt`, `protein/length`, `protein/frac_ge70` over processing steps
3. **Distributions** — image panels for all Category 1-5 plots (populated at end of run)
4. **Protein Table** — the interactive `wandb.Table` for drill-down

### 13.8 CLI Flags for W&B

Add to the script's CLI:

```
--wandb-project TEXT      W&B project name (default: "quality-graft")
--wandb-run-name TEXT     W&B run name (default: auto-generated)
--wandb-entity TEXT       W&B entity/team (default: personal)
--no-wandb                Disable W&B logging entirely
```

When `--no-wandb` is set, all `wandb_logger` functions become no-ops (check `wandb.run is not None` internally).