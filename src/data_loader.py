"""Data loading and preprocessing for dictator game experiments."""

import csv
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CLIP_EPS = 1e-3  # clip sharing ratios away from 0 and 1 for Beta likelihood


def load_dictator_data(filepath=None, game_type=None, clip_eps=CLIP_EPS):
    """Load dictator game data and return sharing ratios.

    Parameters
    ----------
    filepath : str or Path, optional
        Path to CSV file. Defaults to data/dictator_game.csv.
    game_type : str, optional
        Filter by 'giving' or 'taking'. None returns all.
    clip_eps : float
        Clip ratios to [eps, 1-eps] for numerical stability with Beta likelihood.

    Returns
    -------
    data : dict with keys 'sharing_ratio', 'game_type', 'female', 'female_opp'
        Each value is a numpy array.
    """
    if filepath is None:
        filepath = DATA_DIR / "dictator_game.csv"

    ratios, games, females, female_opps = [], [], [], []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if game_type and row["game_type"] != game_type:
                continue
            ratios.append(float(row["sharing_ratio"]))
            games.append(row["game_type"])
            females.append(int(row["female"]))
            female_opps.append(int(row["female_opp"]))

    ratios = np.array(ratios)
    if clip_eps > 0:
        ratios = np.clip(ratios, clip_eps, 1 - clip_eps)

    return {
        "sharing_ratio": ratios,
        "game_type": np.array(games),
        "female": np.array(females),
        "female_opp": np.array(female_opps),
    }


def subsample(data, n, rng=None):
    """Random subsample of sharing ratios.

    Parameters
    ----------
    data : np.ndarray
        Array of sharing ratios.
    n : int
        Subsample size.
    rng : np.random.Generator or int, optional
        Random number generator or seed.

    Returns
    -------
    np.ndarray of shape (n,)
    """
    if isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(rng)
    elif rng is None:
        rng = np.random.default_rng(42)
    idx = rng.choice(len(data), size=n, replace=False)
    return data[idx]


def train_test_split(data, test_frac=0.3, seed=42):
    """Split sharing ratios into train and test sets."""
    rng = np.random.default_rng(seed)
    n = len(data)
    n_test = max(1, int(n * test_frac))
    idx = rng.permutation(n)
    return data[idx[n_test:]], data[idx[:n_test]]
