import torch

from geodiff_gan.spectral_indices import spectral_index_images, spectral_index_metrics


BANDS = ("red", "green", "blue", "nir", "swir1", "swir2")


def test_known_spectral_indices():
    image = torch.tensor([0.2, 0.3, 0.1, 0.6, 0.4, 0.25]).view(1, 6, 1, 1)
    indices = spectral_index_images(image, BANDS)
    assert torch.allclose(indices["ndvi"], torch.tensor([[[0.5]]]))
    assert torch.allclose(indices["ndwi"], torch.tensor([[[-1 / 3]]]))
    assert torch.allclose(indices["ndbi"], torch.tensor([[[-0.2]]]))


def test_index_metrics_measure_improvement_over_lr():
    target = torch.rand(1, 6, 12, 12) * 0.8 + 0.1
    lr = torch.nn.functional.interpolate(target, size=(4, 4), mode="area")
    prediction = target.clone()
    metrics = spectral_index_metrics(
        prediction, target, lr, BANDS, mask=torch.ones(1, 1, 12, 12)
    )
    for name in ("ndvi", "ndwi", "ndbi"):
        assert metrics[f"{name}_mae"] == 0
        assert metrics[f"{name}_mae_improvement_vs_bicubic"] >= 0
        assert metrics[f"{name}_correlation"] > 0.999
