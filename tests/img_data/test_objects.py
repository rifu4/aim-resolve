"""Tests for aim_resolve.img_data.objects — get_masks utility."""

from aim_resolve.img_data.objects import get_masks


class TestGetMasks:
    def test_default_shape(self):
        masks = get_masks()
        assert masks.ndim == 3
        assert masks.shape[1:] == (256, 256)
        # masks.npz has 90 masks; padding adds m_max-90=10 → 100 total
        assert masks.shape[0] == 100

    def test_custom_range(self):
        masks = get_masks(m_min=10, m_max=20)
        assert masks.shape[0] == 11

    def test_m_max_above_90_pads_zeros(self):
        masks = get_masks(m_min=85, m_max=95)
        # 90 original + 5 padded = 95 total; slice [85:96] → 10 items
        assert masks.shape[0] == 10
        # The last entry should be all zeros (from padding)
        assert masks[-1].sum() == 0.0

    def test_values_in_range(self):
        masks = get_masks(m_min=0, m_max=10)
        assert masks.min() >= 0.0
        assert masks.max() <= 1.0
