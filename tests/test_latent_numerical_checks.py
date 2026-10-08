import importlib.util
import unittest
from pathlib import Path

FILE = Path(__file__).resolve().parents[1] / 'baseline5_latentRATT/numerical_checks.py'
spec = importlib.util.spec_from_file_location('latent_numerical_checks', FILE)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


class NumericalChecksTests(unittest.TestCase):
    def test_finite_values_are_preserved(self):
        self.assertEqual(checks.require_finite(-1.25, 'reward'), -1.25)
        self.assertEqual(checks.require_finite(0, 'gradient'), 0.0)

    def test_nonfinite_values_fail(self):
        for value in [float('nan'), float('inf'), -float('inf')]:
            with self.subTest(value=value), self.assertRaises(FloatingPointError):
                checks.require_finite(value, 'loss')

    def test_rl_requires_a_finite_reward(self):
        config = {'numerical_validation': checks.VALIDATION_POLICY, 'stage': 'rl',
                  'completed_steps': 1, 'last_loss': 0.1, 'last_gradient_norm': 0.2}
        with self.assertRaises(ValueError):
            checks.validate_statistics(config)
        config['last_reward'] = -0.5
        checks.validate_statistics(config)
        config['last_reward'] = float('inf')
        with self.assertRaises(FloatingPointError):
            checks.validate_statistics(config)
