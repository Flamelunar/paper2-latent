"""Fail closed on invalid LatentRATT training statistics."""

import math

VALIDATION_POLICY = 'finite_loss_reward_gradient_v1'


def require_finite(value, name):
    if hasattr(value, 'detach'):
        value = value.detach().item()
    numeric = float(value)
    if not math.isfinite(numeric):
        raise FloatingPointError(f'Non-finite {name}: {numeric}')
    return numeric


def validate_statistics(config):
    if config.get('numerical_validation') != VALIDATION_POLICY:
        raise ValueError('Missing finite loss/reward/gradient validation provenance')
    if config.get('completed_steps', 0) > 0:
        for key in ['last_loss', 'last_gradient_norm']:
            if key not in config:
                raise ValueError(f'Missing training statistic: {key}')
            require_finite(config[key], key)
        if config.get('max_steps') == 1 and config['last_gradient_norm'] <= 0:
            raise ValueError('Smoke test must demonstrate a nonzero backward gradient')
        if config.get('stage') == 'rl':
            if 'last_reward' not in config:
                raise ValueError('Missing training statistic: last_reward')
            require_finite(config['last_reward'], 'last_reward')
