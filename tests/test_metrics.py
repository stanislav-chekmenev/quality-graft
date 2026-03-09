"""Tests for pLDDT training metrics."""

import torch
import pytest


class TestPlddtAccuracy:
    """Tests for masked top-1 bin accuracy."""

    def test_perfect_prediction(self):
        from quality_graft.training.metrics import plddt_accuracy

        logits = torch.zeros(2, 5, 50)
        labels = torch.tensor([[0, 1, 2, 3, 4], [10, 20, 30, 40, 49]])
        # Set the correct bin to have highest logit
        for b in range(2):
            for i in range(5):
                logits[b, i, labels[b, i]] = 10.0
        mask = torch.ones(2, 5)
        acc = plddt_accuracy(logits, labels, mask)
        assert abs(acc.item() - 1.0) < 1e-6

    def test_zero_accuracy(self):
        from quality_graft.training.metrics import plddt_accuracy

        logits = torch.zeros(1, 4, 50)
        labels = torch.tensor([[10, 20, 30, 40]])
        # Set wrong bins to highest
        for i in range(4):
            logits[0, i, (labels[0, i].item() + 1) % 50] = 10.0
        mask = torch.ones(1, 4)
        acc = plddt_accuracy(logits, labels, mask)
        assert abs(acc.item()) < 1e-6

    def test_masking(self):
        from quality_graft.training.metrics import plddt_accuracy

        logits = torch.zeros(1, 4, 50)
        labels = torch.tensor([[0, 1, 2, 3]])
        # First two correct, last two wrong
        logits[0, 0, 0] = 10.0
        logits[0, 1, 1] = 10.0
        logits[0, 2, 49] = 10.0
        logits[0, 3, 49] = 10.0
        # Mask out the two wrong ones
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        acc = plddt_accuracy(logits, labels, mask)
        assert abs(acc.item() - 1.0) < 1e-6


class TestPlddtMae:
    """Tests for pLDDT mean absolute error."""

    def test_perfect_prediction(self):
        from quality_graft.training.metrics import plddt_mae

        # Logits that put all probability on the correct bin
        logits = torch.full((1, 3, 50), -100.0)
        labels = torch.tensor([[10, 20, 30]])
        for i in range(3):
            logits[0, i, labels[0, i]] = 100.0
        mask = torch.ones(1, 3)
        mae = plddt_mae(logits, labels, mask)
        assert mae.item() < 1e-4

    def test_masking(self):
        from quality_graft.training.metrics import plddt_mae

        logits = torch.full((1, 2, 50), -100.0)
        labels = torch.tensor([[10, 20]])
        # First residue correct, second totally wrong
        logits[0, 0, 10] = 100.0
        logits[0, 1, 0] = 100.0  # wrong bin
        # Mask out the wrong one
        mask = torch.tensor([[1.0, 0.0]])
        mae = plddt_mae(logits, labels, mask)
        assert mae.item() < 1e-4


class TestPearsonR:
    """Tests for per-protein Pearson correlation."""

    def test_perfect_correlation(self):
        from quality_graft.training.metrics import pearson_r

        pred = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9]])
        target = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9]])
        mask = torch.ones(1, 5)
        r = pearson_r(pred, target, mask)
        assert abs(r.item() - 1.0) < 1e-5

    def test_negative_correlation(self):
        from quality_graft.training.metrics import pearson_r

        pred = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9]])
        target = torch.tensor([[0.9, 0.7, 0.5, 0.3, 0.1]])
        mask = torch.ones(1, 5)
        r = pearson_r(pred, target, mask)
        assert abs(r.item() - (-1.0)) < 1e-5

    def test_batch_averaging(self):
        from quality_graft.training.metrics import pearson_r

        # Two proteins: one perfect, one perfect negative
        pred = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9],
                             [0.1, 0.3, 0.5, 0.7, 0.9]])
        target = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9],
                               [0.9, 0.7, 0.5, 0.3, 0.1]])
        mask = torch.ones(2, 5)
        r = pearson_r(pred, target, mask)
        assert abs(r.item()) < 1e-5  # average of 1.0 and -1.0


class TestSpearmanR:
    """Tests for per-protein Spearman rank correlation."""

    def test_perfect_rank_correlation(self):
        from quality_graft.training.metrics import spearman_r

        pred = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9]])
        target = torch.tensor([[0.2, 0.4, 0.6, 0.8, 1.0]])
        mask = torch.ones(1, 5)
        r = spearman_r(pred, target, mask)
        assert abs(r.item() - 1.0) < 1e-5

    def test_masking(self):
        from quality_graft.training.metrics import spearman_r

        pred = torch.tensor([[0.1, 0.3, 0.5, 999.0, 999.0]])
        target = torch.tensor([[0.2, 0.4, 0.6, -999.0, -999.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
        r = spearman_r(pred, target, mask)
        assert abs(r.item() - 1.0) < 1e-5