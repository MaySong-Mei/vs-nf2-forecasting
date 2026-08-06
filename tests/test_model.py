import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class ModelTests(unittest.TestCase):
    def test_forward_shapes(self):
        import torch

        from ucsd_repro.model import DeepGrowthLite

        model = DeepGrowthLite(latent_channels=8, temporal_order=2, decoder_hidden=16, decoder_layers=3)
        observed = torch.randn(1, 2, 2, 16, 16, 16)
        days = torch.tensor([[0.0, 100.0, 220.0]])
        coords = torch.rand(1, 32, 3) * 2 - 1
        reconstructed, predicted, regularization = model(observed, days, coords)
        self.assertEqual(tuple(reconstructed.shape), (1, 2, 32))
        self.assertEqual(tuple(predicted.shape), (1, 32))
        self.assertEqual(regularization.ndim, 0)


if __name__ == "__main__":
    unittest.main()
