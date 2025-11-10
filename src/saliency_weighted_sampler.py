#!/usr/bin/env python3
"""
Saliency-weighted API sampler.

Usage:
    from saliency_weighted_sampler import Sampler
    s = Sampler(
        api_candidates="checkpoints/api_candidates.json",
        api_effectiveness="checkpoints/api_effectiveness.json",
        benign_counts="checkpoints/benign_api_counts.json",
        alpha=0.7, beta=0.25, gamma=0.05, temp=0.9
    )
    event = s.sample_api_event(idx=3)

Notes:
 - api_candidates: JSON list of api tokens (strings)
 - api_effectiveness: JSON list of {"api": "api:ReadFile", "mean_delta": 0.12, "count": 50}
 - benign_counts: optional JSON map {"api:ReadFile": 123, ...}
 - alpha/beta/gamma: weights for (effectiveness, benign_freq, uniform)
 - temp: temperature for sharpening/flattening the distribution (<1 -> sharper, >1 -> flatter)
 - The sampler also exposes sample_api() to get raw api token.
"""
from pathlib import Path
import json
import random
import math
import numpy as np
from typing import Optional, List, Dict

def load_json(path: Optional[str]):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

class Sampler:
    def __init__(
        self,
        api_candidates: str,
        api_effectiveness: Optional[str] = None,
        benign_counts: Optional[str] = None,
        alpha: float = 0.7,
        beta: float = 0.25,
        gamma: float = 0.05,
        temp: float = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Build sampler probabilities from inputs.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # load api list
        apis = load_json(api_candidates)
        if not isinstance(apis, list):
            raise ValueError(f"api_candidates must be a JSON list, got {type(apis)} from {api_candidates}")
        self.apis: List[str] = apis

        # load effectiveness info -> map api -> mean_delta
        eff_list = load_json(api_effectiveness) or []
        eff_map = {}
        for item in eff_list:
            try:
                key = item.get("api") if isinstance(item, dict) else None
                if key:
                    eff_map[key] = float(item.get("mean_delta", 0.0))
            except Exception:
                continue

        # load benign counts -> map api -> count
        freq_map = load_json(benign_counts) or {}

        # numeric vectors aligned with self.apis
        eff_vec = np.array([eff_map.get(a, 0.0) for a in self.apis], dtype=float)
        freq_vec = np.array([float(freq_map.get(a, 0.0)) for a in self.apis], dtype=float)

        # normalize to [0,1] robustly
        def normalize(vec):
            if vec.size == 0:
                return vec
            mn = vec.min()
            mx = vec.max()
            if mx - mn < 1e-12:
                return np.zeros_like(vec)
            return (vec - mn) / (mx - mn + 1e-12)

        eff_norm = normalize(eff_vec)
        freq_norm = normalize(freq_vec)

        # combine weights
        weights = alpha * eff_norm + beta * freq_norm + gamma
        # clamp
        weights = np.clip(weights, 0.0, None)
        # temperature scaling: treat weights as unnormalized logits
        if temp != 1.0:
            # prevent log(0)
            safe = np.clip(weights, 1e-12, None)
            logits = np.log(safe)
            scaled = np.exp(logits / float(temp))
            weights = scaled

        # ensure non-zero
        if weights.sum() <= 0:
            probs = np.ones_like(weights) / len(weights)
        else:
            probs = weights / float(weights.sum())

        self.probs = probs.tolist()
        # basic sanity: keep a small headroom for pure exploration
        # (re-normalize to mix in pure uniform)
        uniform_p = np.ones_like(probs) / len(probs)
        mix_eps = 0.01
        self.probs = list((1.0 - mix_eps) * np.array(self.probs) + mix_eps * uniform_p)

    # --------------------
    # Sampling helpers
    # --------------------
    def sample_api(self) -> str:
        """Return an api token string (e.g., 'api:ReadFile')."""
        return random.choices(self.apis, weights=self.probs, k=1)[0]

    def sample_event_from_api(self, api_token: str, idx: int = 0) -> str:
        """Return a realistic event string for the chosen API token."""
        api = api_token.split("api:")[-1]
        low = api.lower()

        # common file ops
        if any(x in low for x in ("readfile", "writefile", "createfile", "openfile", "appendfile")):
            return f"api:{api} path:C:\\\\Windows\\\\Temp\\\\pad{idx}.tmp"

        # registry ops
        if low.startswith("reg") or "registry" in low:
            return f"api:{api} path:HKEY_LOCAL_MACHINE\\\\Software\\\\Vendor"

        # network ops
        if any(x in low for x in ("connect", "send", "recv", "socket", "accept")):
            # small variation on IP/port
            octet = (idx * 37) % 250 + 1
            port = 1024 + ((idx * 13) % 5000)
            return f"api:{api} ip:127.0.0.{octet} port:{port}"

        # process/thread
        if "process" in low or "createprocess" in low:
            return f"api:{api} pid:{1000 + (idx % 5000)}"

        if "thread" in low:
            return f"api:{api} tid:{2000 + (idx % 5000)}"

        # fallback concise token
        return f"api:{api}"

    def sample_api_event(self, idx: int = 0) -> str:
        """Sample an API token and return an event string built from it."""
        api = self.sample_api()
        return self.sample_event_from_api(api, idx=idx)

    def topk_by_prob(self, k: int = 10) -> List[str]:
        """Return top-k api tokens by internal probability."""
        idxs = sorted(range(len(self.apis)), key=lambda i: -self.probs[i])[:k]
        return [self.apis[i] for i in idxs]

    # small utility to save current probs for inspection/debug
    def export_probs(self, path: str):
        data = [{"api": a, "prob": float(p)} for a, p in zip(self.apis, self.probs)]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
