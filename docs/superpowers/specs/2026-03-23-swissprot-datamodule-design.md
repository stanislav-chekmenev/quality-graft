# SwissProt DataModule Refactoring — Design Spec

## Problem

Train the distillation head on SwissProt PDB files (~550K AlphaFold-predicted structures). These files:
- Are pre-downloaded at `/mnt/labs/shared/databases/swissprot_pdb_v4/files/`
- Use naming convention `AF-{UniProtID}-F1-model_v4.pdb`
- Store pLDDT scores in the B-factor column (0-100 scale) — no Boltz-1 prediction needed
- Have no `/raw` directory structure and are not preprocessed as `.pt` files

The existing pipeline assumes RCSB PDB structures, CIF format, and Boltz-1 for pLDDT labels. It needs to be extended, not modified.

## Approach

Subclass existing classes (Approach A). No La-Proteina code changes.

## Components

### 1. SwissProtDataSelector

**File:** `src/quality_graft/data/swissprot_selector.py`
**Class:** `SwissProtDataSelector(PDBDataSelector)`

**Constructor params** (in addition to inherited):
- `source_dir: str` — path to shared SwissProt PDB directory
- `metadata_tsv: str` — path to UniProt TSV file (accession + length)
- `alphafold_version: int = 4` — for filename pattern matching

**`create_dataset()` override:**
1. Load UniProt TSV (`accession`, `length`) into a DataFrame
2. Apply filters against the DataFrame:
   - `min_length` / `max_length` → filter on the `length` column
   - `fraction` → `df.sample(frac=...)`
   - `exclude_ids` / `exclude_ids_from_file` → remove matching accessions
3. Build expected filenames: `AF-{accession}-F1-model_v{version}.pdb`
4. Cross-reference against files actually present in `source_dir` — keep only rows where the file exists
5. Return DataFrame with columns: `pdb` (filename stem, e.g. `AF-A0A009IHW8-F1-model_v4`), `id` (same), `accession` (UniProt ID), `length`

No PDBManager, no RCSB queries, no download. Pure metadata + filesystem check.

### 2. SwissProtDataModule

**File:** `src/quality_graft/data/swissprot_datamodule.py`
**Class:** `SwissProtDataModule(QualityGraftDataModule)`

**Constructor params** (in addition to inherited):
- `source_dir: str` — shared SwissProt directory path

`boltz_config` is inherited but ignored — no Boltz pass runs.

**`prepare_data()` override:**
1. Call `self.dataselector.create_dataset()` to get filtered DataFrame
2. Skip download — files are already in `raw/` (copied by the separate script)
3. Call `_process_structure_data()` (inherited) to convert PDB → PyG `.pt` files
4. No Boltz pass — pLDDT is extracted during processing
5. Save the filtered DataFrame CSV

**`_load_and_process_pdb()` override:**
Extends the parent method. After parent creates the PyG graph (which has `graph.bfactor` from `store_bfactor=True`), adds:
```python
graph.plddt = graph.bfactor_avg / 100.0        # B-factor is pLDDT on 0-100 scale
graph.plddt_bin = plddt_to_bin(graph.plddt)     # bin to 0..49
graph.plddt_logits = None                       # no logits (hard targets only)
```

Every `.pt` file comes out of processing already labeled. The `plddt_status.csv` tracking is still written so `setup()` filtering works unchanged.

**`setup()` and `_get_dataset()`** — inherited as-is.

**File naming convention:** `AF-A0A009IHW8-F1-model_v4` as the `pdb` code throughout. Raw: `AF-A0A009IHW8-F1-model_v4.pdb`, processed: `AF-A0A009IHW8-F1-model_v4.pt`.

### 3. Copy Script

**File:** `scripts/copy_swissprot.py`

**Interface:**
```bash
python scripts/copy_swissprot.py \
  --source-dir /mnt/labs/shared/databases/swissprot_pdb_v4/files \
  --dest-dir /scratch/schekmenev/swissprot_v4/raw \
  --metadata-tsv data/swissprot/uniprot_metadata.tsv \
  --min-length 30 \
  --max-length 512 \
  --fraction 1.0 \
  --exclude-ids-file data/swissprot/exclude.txt  # optional
```

**Flow:**
1. Instantiate `SwissProtDataSelector` with the provided filter params
2. Call `create_dataset()` → filtered DataFrame with filenames
3. Scan `dest-dir` for already-copied files
4. Copy only missing files (`shutil.copy2`) with a progress bar
5. Save the filtered file list to `dest-dir/../filtered_ids.txt`
6. Print summary: N total filtered, N already present, N copied

**Idempotent:** Running again only copies files that don't exist yet. No overwrites.

**sbatch integration:** Called in `scripts/train_full.sbatch` before the training command:
```bash
python scripts/copy_swissprot.py --source-dir ... --dest-dir $SCRATCH_DIR/raw ...
python scripts/train.py data=swissprot data.data_dir=$SCRATCH_DIR ...
```

### 4. Config & Train Script Integration

**New config:** `configs/data/swissprot.yaml`
```yaml
data_dir: data/swissprot/
source_dir: /mnt/labs/shared/databases/swissprot_pdb_v4/files
metadata_tsv: data/swissprot/uniprot_metadata.tsv
alphafold_version: 4
max_length: ${training.max_length}
min_length: ${training.min_length}
format: pdb
local_only: false
num_plddt_bins: 50
train_val_test: [0.8, 0.15, 0.05]
batch_size: ${training.batch_size}
num_workers: ${training.num_workers}
fraction: 1.0
exclude_ids: null
exclude_ids_from_file: null
selector_num_workers: 32
database: swissprot
```

**`scripts/train.py` `build_data_module()` changes:**
- Check `data_cfg.get("database", "pdb")`
- If `"swissprot"`: instantiate `SwissProtDataSelector` + `SwissProtDataModule`
- If `"pdb"`: existing path (unchanged)

**One-time metadata download:** `scripts/download_uniprot_tsv.py`
- Fetches `https://rest.uniprot.org/uniprotkb/stream?format=tsv&query=(reviewed:true)&fields=accession,length`
- Saves to `data/swissprot/uniprot_metadata.tsv`
- Run once manually

## Data Flow Summary

```
UniProt TSV + source_dir scan
        │
        ▼
SwissProtDataSelector.create_dataset()
        │  (filters: length, fraction, exclude_ids, file existence)
        ▼
    Filtered DataFrame
        │
        ├──► copy_swissprot.py ──► copies to $SCRATCH/raw/
        │
        ▼
SwissProtDataModule.prepare_data()
        │  (no download, no Boltz)
        ▼
_load_and_process_pdb()  ──► .pt files with plddt from B-factor
        │
        ▼
    setup() / split / train
```

## Key Design Decisions

- **No La-Proteina changes.** All new code is in `src/quality_graft/data/` and `scripts/`.
- **pLDDT from B-factors.** AlphaFold stores pLDDT as B-factor (0-100). Divided by 100 to match existing 0-1 scale. Binned with existing `plddt_to_bin()`.
- **Hard targets only.** `graph.plddt_logits = None` — the training loss must handle this (cross-entropy on `plddt_bin` instead of KL on logits).
- **Idempotent copy.** The copy script only transfers missing files, safe to re-run.
- **Format is PDB, not CIF.** `_prepare_boltz_yaml()` (CIF parsing) is never called.

## Files Created/Modified

| File | Action |
|---|---|
| `src/quality_graft/data/swissprot_selector.py` | **New** — `SwissProtDataSelector` |
| `src/quality_graft/data/swissprot_datamodule.py` | **New** — `SwissProtDataModule` |
| `scripts/copy_swissprot.py` | **New** — filter + copy script |
| `scripts/download_uniprot_tsv.py` | **New** — one-time TSV download |
| `configs/data/swissprot.yaml` | **New** — Hydra config |
| `scripts/train.py` | **Modified** — `build_data_module()` branching on `database` |
