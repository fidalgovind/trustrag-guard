"""Layer 1 fitted detectors: is a chunk unusual for this corpus?

Signal A compares a chunk with the query. These compare a chunk with the clean corpus
itself: Isolation Forest, and Mahalanobis distance after PCA with a Ledoit-Wolf
covariance. Both are fitted offline on clean passages and only score online.

They failed the separation test, so they are not in
fusion. Published poison reads like the corpus, so it is not an outlier.

Known limits: a fitted model describes one corpus and one embedder, so it does not
transfer. Published poison is short, so a score may reflect length more than content.
"""
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

DEFAULT_TREES = 100
DEFAULT_PCA_DIMS = 40


@dataclass
class FittedDetectors:
    forest: IsolationForest
    pca: PCA
    precision: np.ndarray    # inverse covariance in PCA space
    mean: np.ndarray         # mean in PCA space


def fit_detectors(clean_embs, trees=DEFAULT_TREES, dims=DEFAULT_PCA_DIMS, seed=42):
    """Offline step, on clean passages only."""
    x = np.asarray(clean_embs, dtype=np.float64)
    forest = IsolationForest(n_estimators=trees, random_state=seed).fit(x)
    pca = PCA(n_components=dims, random_state=seed).fit(x)
    z = pca.transform(x)
    cov = LedoitWolf().fit(z)
    return FittedDetectors(forest, pca, cov.precision_, cov.location_)


def isolation_score(fitted, chunk_embs):
    """Online step: higher is more unusual."""
    return -fitted.forest.score_samples(np.asarray(chunk_embs, dtype=np.float64))


def mahalanobis_score(fitted, chunk_embs):
    """Online step: squared distance from the clean centre. Higher is more unusual."""
    z = fitted.pca.transform(np.asarray(chunk_embs, dtype=np.float64)) - fitted.mean
    return np.einsum("ij,jk,ik->i", z, fitted.precision, z)


def save(fitted, path):
    joblib.dump(fitted, path)


def load(path):
    return joblib.load(path)
