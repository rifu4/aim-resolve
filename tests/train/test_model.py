"""Tests for aim_resolve.train.model — SegmentationModel.

These tests require PyTorch Lightning and segmentation_models_pytorch.
They are skipped if those dependencies are not available.
"""

import pytest

try:
    import lightning  # noqa: F401
    import segmentation_models_pytorch  # noqa: F401
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False

pytestmark = pytest.mark.skipif(not HAS_LIGHTNING, reason="lightning not importable")


@pytest.fixture
def seg_model():
    from aim_resolve.train.model import SegmentationModel
    return SegmentationModel.build(
        arch="Unet",
        model_args=dict(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=1,
            classes=2,
        ),
        loss="bce",
        optimizer=dict(lr=1e-3),
        scheduler=dict(T_max=100),
    )


class TestSegmentationModelBuild:
    def test_build(self, seg_model):
        from aim_resolve.train.model import SegmentationModel
        assert isinstance(seg_model, SegmentationModel)

    def test_unknown_loss_raises(self):
        from aim_resolve.train.model import SegmentationModel
        with pytest.raises(ValueError, match="Unknown loss"):
            SegmentationModel.build(
                arch="Unet",
                model_args=dict(encoder_name="resnet18", encoder_weights=None, in_channels=1, classes=2),
                loss="unknown",
                optimizer=dict(lr=1e-3),
                scheduler=dict(T_max=100),
            )


class TestSegmentationModelForward:
    def test_forward(self, seg_model):
        import torch
        seg_model.eval()
        x = torch.randn(1, 1, 32, 32)
        out = seg_model.forward(x)
        assert out.shape == (1, 2, 32, 32)

    def test_forward_sigmoid(self, seg_model):
        import torch
        seg_model.eval()
        x = torch.randn(1, 1, 32, 32)
        out = seg_model.forward_sigmoid(x)
        assert out.dtype == torch.bool


class TestSegmentationModelSaveLoad:
    def test_roundtrip(self, seg_model, tmp_path):
        seg_model.save("test_model", odir=str(tmp_path))
        from aim_resolve.train.model import SegmentationModel
        loaded = SegmentationModel.load("test_model", odir=str(tmp_path))
        assert isinstance(loaded, SegmentationModel)
