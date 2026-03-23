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

**Constructor:** Calls `super().__init__()` with all PDB-specific params at their defaults (`molecule_type=None`, `experiment_types=None`, `oligomeric_min=None`, `oligomeric_max=None`, `best_resolution=None`, `worst_resolution=None`, `has_ligands=None`, `remove_ligands=None`, `remove_non_standard_residues=False`, `remove_pdb_unavailable=False`, `labels=None`, `remove_cath_unavailable=False`). Forwards `data_dir`, `fraction`, `min_length`, `max_length`, `exclude_ids`, `exclude_ids_from_file`, `num_workers` to the parent.

**Additional constructor params:**
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

**Note:** The DataFrame intentionally has no `chain` column — AlphaFold structures are single-chain. Downstream code must call `_process_structure_data(pdb_codes, chains=None)` (not `chains=df["chain"].tolist()`).

**Note:** The DataFrame has no `sequence` column. Sequence-similarity splitting is not supported for SwissProt — only random splitting.

### 2. SwissProtDataModule

**File:** `src/quality_graft/data/swissprot_datamodule.py`
**Class:** `SwissProtDataModule(QualityGraftDataModule)`

**Constructor params** (in addition to inherited):
- `source_dir: str` — shared SwissProt directory path

`boltz_config` is accepted by the constructor (inherited from `QualityGraftDataModule`) but is never used — no Boltz pass runs. The `build_data_module()` function passes an empty dict `{}` for SwissProt.

**`prepare_data()` override:**
This method **completely replaces** the parent chain — it must NOT call `super().prepare_data()` (which would trigger Boltz-1 prediction from `QualityGraftDataModule`).

Steps:
1. Call `self.dataselector.create_dataset()` to get filtered DataFrame. If zero results, raise `ValueError`.
2. Skip download — files are already in `raw/` (copied by the separate script)
3. Call `self._process_structure_data(df_data["pdb"].tolist(), chains=None)` — explicitly pass `chains=None` since AlphaFold structures are single-chain and the DataFrame has no `chain` column
4. No Boltz pass — pLDDT is extracted during processing
5. Save the filtered DataFrame CSV using `_get_file_identifier()` for the filename
6. **Write `plddt_status.csv`**: After processing, iterate over all successfully created `.pt` files and write one row per structure with `has_plddt=true`. This is critical — the inherited `setup()` filters splits against this CSV, so without it all structures would be filtered out.
7. Log summary: N structures processed, N failed (parse errors are silently skipped and logged as warnings by the inherited `_process_structure_data`).

**`_load_and_process_pdb()` override:**
Copies the parent method body (from `PDBLightningDataModule._load_and_process_pdb`) rather than calling `super()` — this avoids double I/O (save + reload + re-save) which matters at 550K scale. The copied method is identical except:
1. After `graph.bfactor_avg = torch.mean(graph.bfactor, dim=-1)`, adds:
```python
graph.plddt = graph.bfactor_avg / 100.0        # B-factor is pLDDT on 0-100 scale
graph.plddt_bin = plddt_to_bin(graph.plddt)     # bin to 0..49
graph.plddt_logits = None                       # no logits (hard targets only)
graph.database = "swissprot"                    # distinguish from PDB data source
```
2. The `torch.save(graph, ...)` at the end saves the fully labeled graph in one pass.

Every `.pt` file comes out of processing already labeled — no second pass needed.

**Maintenance note:** The copied `_load_and_process_pdb` body creates a coupling with the parent in `pdb_data.py` (lines ~628-704). If the parent method changes, the SwissProt copy may silently diverge. The implementation should include a comment referencing the parent method source.

**`_get_file_identifier(self, ds)` override:**
Must accept the `ds` parameter to match the parent signature (called as `self._get_file_identifier(self.dataselector)` from both `prepare_data()` and `setup()`). Returns a SwissProt-specific string:
```python
def _get_file_identifier(self, ds):
    return f"df_swissprot_f{ds.fraction}_minl{ds.min_length}_maxl{ds.max_length}"
```
Avoids the parent's long string with many `None` PDB-specific values.

**`setup()` and `_get_dataset()`** — inherited as-is from `QualityGraftDataModule`. The pLDDT filtering in `setup()` works because `plddt_status.csv` is populated during `prepare_data()`.

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
split_type: random  # sequence_similarity not supported (no sequence column)
database: swissprot
```

**`scripts/train.py` `build_data_module()` changes:**

Branch on `data_cfg.get("database", "pdb")`. If `"pdb"`: existing path (unchanged). If `"swissprot"`:

```python
if database == "swissprot":
    dataselector = SwissProtDataSelector(
        data_dir=data_cfg.data_dir,
        source_dir=data_cfg.source_dir,
        metadata_tsv=data_cfg.metadata_tsv,
        alphafold_version=data_cfg.get("alphafold_version", 4),
        fraction=data_cfg.get("fraction", 1.0),
        min_length=data_cfg.min_length,
        max_length=data_cfg.max_length,
        exclude_ids=data_cfg.get("exclude_ids", None),
        exclude_ids_from_file=data_cfg.get("exclude_ids_from_file", None),
        num_workers=data_cfg.get("selector_num_workers", 32),
    )
    datasplitter = PDBDataSplitter(
        data_dir=data_cfg.data_dir,
        train_val_test=list(data_cfg.train_val_test),
    )
    transforms = [
        TransformWrapper(lp_transforms.CoordsToNanometers),
        TransformWrapper(lp_transforms.CenterStructureTransform),
    ]
    return SwissProtDataModule(
        data_dir=data_cfg.data_dir,
        source_dir=data_cfg.source_dir,
        dataselector=dataselector,
        datasplitter=datasplitter,
        format="pdb",
        boltz_config={},  # empty — never used but required by parent
        num_plddt_bins=data_cfg.num_plddt_bins,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        transforms=transforms,
    )
```

Note: the same `CoordsToNanometers` and `CenterStructureTransform` transforms are applied as in the PDB path.

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
- **Hard targets only.** `graph.plddt_logits = None`. The existing `_compute_loss()` in `QualityGraftLightningModule` already handles `teacher_logits=None` by falling back to pure cross-entropy (line 150-151 of `lightning_module.py`). No loss code changes needed.
- **Idempotent copy.** The copy script only transfers missing files, safe to re-run.
- **Format is PDB, not CIF.** `_prepare_boltz_yaml()` (CIF parsing) is never called.

## Known Limitations & Edge Cases

- **`local_only=True` not supported for SwissProt.** The inherited `_setup_local_only()` splits filenames on `_` to extract pdb/chain, which misparses SwissProt names (e.g. `AF-A0A009IHW8-F1-model_v4` → `chain="v4"`). The config sets `local_only: false`. If local-only mode is needed later, `_setup_local_only()` must be overridden.
- **`overwrite` flag.** The inherited `overwrite` flag is respected by `_process_structure_data()` (which skips existing `.pt` files). The `prepare_data()` override also checks if the DataFrame CSV already exists and skips re-processing if `overwrite=False`.
- **`in_memory=False` always.** At 550K structures, `in_memory=True` would exhaust memory. The config does not set it (defaults to `False`).
- **Parse failures.** PDB files that fail during `protein_to_pyg` conversion are silently skipped (logged as warnings). These structures will not appear in `plddt_status.csv` and will be excluded from training.

## Files Created/Modified

| File | Action |
|---|---|
| `src/quality_graft/data/swissprot_selector.py` | **New** — `SwissProtDataSelector` |
| `src/quality_graft/data/swissprot_datamodule.py` | **New** — `SwissProtDataModule` |
| `scripts/copy_swissprot.py` | **New** — filter + copy script |
| `scripts/download_uniprot_tsv.py` | **New** — one-time TSV download |
| `configs/data/swissprot.yaml` | **New** — Hydra config |
| `scripts/train.py` | **Modified** — `build_data_module()` branching on `database` |
