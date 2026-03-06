"""Tests for the dataset generation pipeline modules."""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pytest
import torch
import yaml

from quality_graft.data.plddt_utils import NUM_PLDDT_BINS, bin_to_plddt, plddt_to_bin
from quality_graft.data.cif_utils import ChainInfo, chains_to_boltz_yaml, parse_cif_chains
from quality_graft.data.boltz_runner import (
    BoltzResult,
    build_boltz_command,
    find_confidence_json,
    find_plddt_npz,
)
from quality_graft.data.wandb_logger import (
    compute_protein_metrics,
    count_segments_below,
    finish_wandb_run,
    init_wandb_run,
    log_protein_metrics,
    longest_contiguous_below,
)


# ── Phase A: pLDDT Binning ──────────────────────────────────────────────────


class TestPlddtBinning:
    def test_boundary_values(self):
        plddt = torch.tensor([0.0, 0.5, 0.99, 1.0])
        bins = plddt_to_bin(plddt)
        assert bins[0] == 0
        assert bins[1] == 25
        assert bins[2] == 49
        assert bins[3] == 49  # clamped

    def test_round_trip(self):
        plddt = torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99])
        bins = plddt_to_bin(plddt)
        recovered = bin_to_plddt(bins)
        assert torch.allclose(plddt, recovered, atol=0.01 + 1e-6)

    def test_shape_preserved(self):
        plddt = torch.rand(4, 10)
        bins = plddt_to_bin(plddt)
        assert bins.shape == (4, 10)

    def test_dtype(self):
        bins = plddt_to_bin(torch.tensor([0.5]))
        assert bins.dtype == torch.int64

    def test_negative_clamped(self):
        plddt = torch.tensor([-0.1, -1.0])
        bins = plddt_to_bin(plddt)
        assert (bins == 0).all()


# ── Phase A: CIF Parsing ───────────────────────────────────────────────────


class TestCifParsing:
    def test_parse_1ubq(self):
        cif_path = PROJECT_ROOT / "data" / "1ubq.cif"
        if not cif_path.exists():
            pytest.skip("data/1ubq.cif not found")
        chains = parse_cif_chains(cif_path)
        assert len(chains) == 1
        assert chains[0].chain_id == "A"
        assert chains[0].n_residues == 76
        assert chains[0].sequence.startswith("MQIFVKTLTG")

    def test_no_protein_chains_raises(self, tmp_path):
        cif_content = (
            "data_empty\n"
            "_entry.id empty\n"
            "loop_\n"
            "_atom_site.group_PDB\n"
            "_atom_site.id\n"
            "_atom_site.type_symbol\n"
            "_atom_site.label_atom_id\n"
            "_atom_site.label_alt_id\n"
            "_atom_site.label_comp_id\n"
            "_atom_site.label_asym_id\n"
            "_atom_site.label_entity_id\n"
            "_atom_site.label_seq_id\n"
            "_atom_site.pdbx_PDB_ins_code\n"
            "_atom_site.Cartn_x\n"
            "_atom_site.Cartn_y\n"
            "_atom_site.Cartn_z\n"
            "_atom_site.occupancy\n"
            "_atom_site.B_iso_or_equiv\n"
            "_atom_site.pdbx_formal_charge\n"
            "_atom_site.auth_seq_id\n"
            "_atom_site.auth_comp_id\n"
            "_atom_site.auth_asym_id\n"
            "_atom_site.auth_atom_id\n"
            "_atom_site.pdbx_PDB_model_num\n"
            "HETATM 1 O O . HOH A 1 1 ? 0.000 0.000 0.000 1.00 0.00 ? 1 HOH A O 1\n"
        )
        cif_file = tmp_path / "empty.cif"
        cif_file.write_text(cif_content)
        with pytest.raises(ValueError, match="No protein chains found"):
            parse_cif_chains(cif_file)

    def test_yaml_generation(self):
        chains = [
            ChainInfo(chain_id="A", sequence="MQIFVKTLTG", n_residues=10),
            ChainInfo(chain_id="B", sequence="MADQLTEEQI", n_residues=10),
        ]
        yaml_str = chains_to_boltz_yaml(chains)
        assert "version: 1" in yaml_str
        assert "msa: empty" in yaml_str
        assert "MQIFVKTLTG" in yaml_str
        assert "MADQLTEEQI" in yaml_str
        parsed = yaml.safe_load(yaml_str)
        assert "sequences" in parsed
        assert "version" in parsed
        assert len(parsed["sequences"]) == 2

    def test_yaml_single_sequence_mode(self):
        chains = [ChainInfo(chain_id="A", sequence="ACGT", n_residues=4)]
        yaml_str = chains_to_boltz_yaml(chains, use_msa=False)
        assert "msa: empty" in yaml_str

    def test_yaml_msa_mode(self):
        chains = [ChainInfo(chain_id="A", sequence="ACGT", n_residues=4)]
        yaml_str = chains_to_boltz_yaml(chains, use_msa=True)
        assert "msa: empty" not in yaml_str


# ── Phase B: Boltz Runner ───────────────────────────────────────────────────


class TestBoltzRunner:
    def test_command_construction_default(self):
        cmd = build_boltz_command(
            yaml_path=Path("/tmp/test.yaml"),
            out_dir=Path("/tmp/output"),
        )
        assert cmd[0] == "boltz"
        assert cmd[1] == "predict"
        assert str(Path("/tmp/test.yaml")) in cmd
        assert "--out_dir" in cmd
        assert "--model" in cmd
        assert "boltz1" in cmd
        assert "--use_msa_server" not in cmd
        assert "--override" not in cmd

    def test_command_construction_with_msa(self):
        cmd = build_boltz_command(
            yaml_path=Path("/tmp/test.yaml"),
            out_dir=Path("/tmp/output"),
            use_msa_server=True,
        )
        assert "--use_msa_server" in cmd

    def test_command_construction_with_override(self):
        cmd = build_boltz_command(
            yaml_path=Path("/tmp/test.yaml"),
            out_dir=Path("/tmp/output"),
            override=True,
        )
        assert "--override" in cmd

    def test_command_construction_custom_params(self):
        cmd = build_boltz_command(
            yaml_path=Path("/tmp/test.yaml"),
            out_dir=Path("/tmp/output"),
            model="boltz1",
            devices=4,
            accelerator="cpu",
            diffusion_samples=5,
            sampling_steps=100,
            recycling_steps=5,
        )
        assert cmd[cmd.index("--model") + 1] == "boltz1"
        assert cmd[cmd.index("--devices") + 1] == "4"
        assert cmd[cmd.index("--accelerator") + 1] == "cpu"
        assert cmd[cmd.index("--diffusion_samples") + 1] == "5"
        assert cmd[cmd.index("--sampling_steps") + 1] == "100"
        assert cmd[cmd.index("--recycling_steps") + 1] == "5"

    def test_npz_path_resolution(self, tmp_path):
        pdb_id = "1ubq"
        pred_dir = tmp_path / "predictions" / pdb_id
        pred_dir.mkdir(parents=True)
        npz_file = pred_dir / f"plddt_{pdb_id}_model_0.npz"
        np.savez(npz_file, plddt=np.array([0.5, 0.6, 0.7]))
        found = find_plddt_npz(tmp_path, pdb_id)
        assert found is not None
        assert found == npz_file

    def test_npz_path_alternate_layout(self, tmp_path):
        pdb_id = "2abc"
        pred_dir = tmp_path / pdb_id / "predictions" / pdb_id
        pred_dir.mkdir(parents=True)
        npz_file = pred_dir / f"plddt_{pdb_id}_model_0.npz"
        np.savez(npz_file, plddt=np.array([0.8, 0.9]))
        found = find_plddt_npz(tmp_path, pdb_id)
        assert found is not None
        assert found == npz_file

    def test_npz_path_boltz_results_layout(self, tmp_path):
        pdb_id = "1ubq"
        pred_dir = tmp_path / f"boltz_results_{pdb_id}" / "predictions" / pdb_id
        pred_dir.mkdir(parents=True)
        npz_file = pred_dir / f"plddt_{pdb_id}_model_0.npz"
        np.savez(npz_file, plddt=np.array([0.5, 0.6]))
        found = find_plddt_npz(tmp_path, pdb_id)
        assert found is not None
        assert found == npz_file

    def test_npz_path_not_found(self, tmp_path):
        found = find_plddt_npz(tmp_path, "nonexistent")
        assert found is None

    def test_confidence_json_resolution(self, tmp_path):
        pdb_id = "1ubq"
        pred_dir = tmp_path / "predictions" / pdb_id
        pred_dir.mkdir(parents=True)
        json_file = pred_dir / f"confidence_{pdb_id}_model_0.json"
        json_file.write_text('{"plddt": 0.85}')
        found = find_confidence_json(tmp_path, pdb_id)
        assert found is not None
        assert found == json_file

    def test_confidence_json_alternate_layout(self, tmp_path):
        pdb_id = "3xyz"
        pred_dir = tmp_path / pdb_id / "predictions" / pdb_id
        pred_dir.mkdir(parents=True)
        json_file = pred_dir / f"confidence_{pdb_id}_model_0.json"
        json_file.write_text('{"plddt": 0.72}')
        found = find_confidence_json(tmp_path, pdb_id)
        assert found is not None
        assert found == json_file

    def test_confidence_json_not_found(self, tmp_path):
        found = find_confidence_json(tmp_path, "nonexistent")
        assert found is None

    def test_boltz_result_dataclass(self):
        result = BoltzResult(
            structure_id="1ubq",
            plddt=np.array([0.5, 0.6]),
            confidence_json={"plddt": 0.55},
            success=True,
            error_msg=None,
        )
        assert result.success
        assert result.structure_id == "1ubq"
        assert result.plddt is not None
        assert len(result.plddt) == 2
        assert result.error_msg is None

    def test_boltz_result_failure(self):
        result = BoltzResult(
            structure_id="bad",
            plddt=None,
            confidence_json=None,
            success=False,
            error_msg="Something went wrong",
        )
        assert not result.success
        assert result.plddt is None
        assert result.error_msg == "Something went wrong"


# ── Phase C: Segment Analysis ───────────────────────────────────────────────


class TestSegmentAnalysis:
    def test_longest_contiguous_below(self):
        plddt = np.array([0.8, 0.3, 0.2, 0.4, 0.9, 0.1])
        assert longest_contiguous_below(plddt, 0.5) == 3

    def test_longest_all_above(self):
        plddt = np.array([0.8, 0.9, 0.7, 0.6])
        assert longest_contiguous_below(plddt, 0.5) == 0

    def test_longest_all_below(self):
        plddt = np.array([0.1, 0.2, 0.3, 0.4])
        assert longest_contiguous_below(plddt, 0.5) == 4

    def test_count_segments_below(self):
        plddt = np.array([0.8, 0.3, 0.2, 0.4, 0.9, 0.1])
        assert count_segments_below(plddt, 0.5) == 2

    def test_count_no_segments(self):
        plddt = np.array([0.8, 0.9, 0.7])
        assert count_segments_below(plddt, 0.5) == 0


# ── Phase C: W&B Logger ─────────────────────────────────────────────────────


class TestWandbLogger:
    def test_no_wandb_mode(self):
        args = argparse.Namespace(no_wandb=True)
        init_wandb_run(args)
        finish_wandb_run()

    def test_protein_metrics_computation(self):
        plddt = np.array([0.95, 0.85, 0.45, 0.30, 0.72])
        metrics = compute_protein_metrics("test", plddt, 5, 1.0)
        assert metrics["protein/structure_id"] == "test"
        assert metrics["protein/length"] == 5
        assert abs(metrics["protein/mean_plddt"] - 0.654) < 1e-6
        assert metrics["protein/frac_ge90"] == pytest.approx(0.2)
        assert metrics["protein/frac_ge70"] == pytest.approx(0.6)
        assert metrics["protein/frac_lt50"] == pytest.approx(0.4)

    def test_log_protein_metrics_no_wandb(self):
        plddt = np.array([0.5, 0.6, 0.7])
        metrics = log_protein_metrics("test", plddt, 3, 1.0, 1, 0, 0)
        assert "protein/mean_plddt" in metrics
        assert "progress/n_processed" in metrics
