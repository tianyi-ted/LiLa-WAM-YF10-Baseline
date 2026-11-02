"""Inference-only, normalized action-tail priors. No model or trainer changes."""
import math
import torch


class TailActionPrior:
    def __init__(self, enabled=False, sigma=1.0, strength=1.0):
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError('reuse sigma must be finite and nonnegative')
        if not math.isfinite(strength) or strength < 0:
            raise ValueError('reuse strength must be finite and nonnegative')
        self.enabled, self.sigma, self.strength = enabled, sigma, strength
        self.reset()

    def reset(self):
        self.previous = None
        self.executed = 0
        self.last_overlap = 0

    def source(self, noise):
        self.last_overlap = 0
        if not self.enabled or self.previous is None:
            return noise
        if noise.shape != self.previous.shape:
            raise ValueError('cached chunk shape changed; reset the episode first')
        if not 0 < self.executed <= noise.shape[1]:
            raise ValueError('a cached plan must advance before replanning')
        count = noise.shape[1] - self.executed
        self.last_overlap = count
        if count == 0:
            return noise
        x = noise.clone()
        x[:, :count] = (self.strength * self.previous[:, self.executed:]
                       + self.sigma * noise[:, :count])
        return x

    def remember(self, normalized_chunk):
        if self.enabled:
            self.previous = normalized_chunk.detach().clone()
            self.executed = 0

    def advance(self):
        if self.enabled:
            self.executed += 1
