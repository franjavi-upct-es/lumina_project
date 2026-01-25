# backend/ml_engine/training/purged_cv.py
"""
Purged Cross-Validation for time series to prevent data leakage
Implements techniques from "Advances in Financial Machine Learning" by Marcos López de Prado
"""

from collections.abc import Generator

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import KFold


class PurgedKFold:
    """
    Purged K-Fold cross-validation for time series

    Purges samples that overlap with the test set to prevent look-ahead bias.
    Also implements embargo period to account for serial correlation.

    Key features:
    - Purging: Remove training samples that overlap with test period
    - Embargo: Add gap after test period before using data for training
    - Sample weighting: Account for overlapping labels
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01, purge_gap: int = 0):
        """
        Initialize Purged K-Fold

        Args:
            n_splits: Number of folds
            embargo_pct: Percentage of samples after test set
            purge_gap: Number of samples to purge before and after test set
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_gap = purge_gap

        logger.info(
            f"Initialized PurgedKFold with {n_splits} splits, "
            f"embargo={embargo_pct:.2%}, purge_gap={purge_gap}"
        )

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        sample_times: pd.Series | None = None,
        pred_times: pd.Series | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generates indices for train/test splits with purging

        Args:
            X: Features array
            y: Target array (not used, kept for sklearn compatibility)
            sample_times: Series with sample start times (index = sample index)
            pred_times: Series with prediction end times (when label is known)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        if sample_times is None or pred_times is None:
            logger.warning("No time information provided, using standard K-Fold")
            kfold = KFold(n_splits=self.n_splits)
            for train_idx, test_idx in kfold.split(X):
                yield train_idx, test_idx
            return

        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate test set size
        test_size = n_samples // self.n_splits

        for fold in range(self.n_splits):
            # Define test set
            test_start = fold * test_size
            test_end = test_start + test_size if fold < self.n_splits - 1 else n_samples
            test_indices = indices[test_start:test_end]

            # Get time range for test set
            test_times = sample_times.iloc[test_indices]
            test_start_time = test_times.min()
            test_end_time = pred_times.iloc[test_indices].max()

            # Purge training samples that overlap with test period
            train_indices = self._purge_overlapping(
                indices, test_indices, sample_times, pred_times, test_start_time, test_end_time
            )

            # Apply embargo - remove samples immediately after test set
            train_indices = self._apply_embargo(
                train_indices,
                test_indices,
                n_samples,
            )

            logger.debug(
                f"Fold {fold + 1}/{self.n_splits}: "
                f"{len(train_indices)} train, {len(test_indices)} test samples"
            )

            yield train_indices, test_indices

    def _purge_overlapping(
        self,
        indices: np.ndarray,
        test_indices: np.ndarray,
        sample_times: pd.Series,
        pred_times: pd.Series,
        test_start_time: pd.Timestamp,
        test_end_time: pd.Timestamp,
    ) -> np.ndarray:
        """
        Remove training samples that overlap with test period

        A training sample overlaps if:
        - Its label time (pred_time) is within test period
        - Its sample time + purge_gap overlaps with test start
        """
        # Start with all indices except test
        mask = ~np.isin(indices, test_indices)
        train_candidates = indices[mask]

        # Remove samples where prediction time overlaps with test periods
        overlap_mask = np.ones(len(train_candidates), dtype=bool)

        for i, idx in enumerate(train_candidates):
            pred_time = pred_times.iloc[idx]
            sample_time = sample_times.iloc[idx]

            # Check if prediction time is during test period
            if test_start_time <= pred_time <= test_end_time:
                overlap_mask[i] = False
                continue

            # Check purge gap before test period
            if self.purge_gap > 0:
                # Convert purge_gap to time delta (assuming sorted by time)
                idx_in_series = sample_times.index.get_loc(sample_times.index[idx])
                test_start_idx = sample_times.index.get_loc(
                    sample_times[sample_times >= test_start_time].index[0]
                )

                if test_start_idx - self.purge_gap <= idx_in_series < test_start_idx:
                    overlap_mask[i] = False

        return train_candidates[overlap_mask]

    def _apply_embargo(
        self,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """
        Apply embargo period after test set

        Removes training samples within embargo_pct of dataset size
        immediately after the test set
        """
        if self.embargo_pct <= 0:
            return train_indices

        # Calculate embargo size
        embargo_size = int(n_samples * self.embargo_pct)

        # Get test set end
        test_end = test_indices.max()

        # Remove samples in embargo period
        embargo_start = test_end + 1
        embargo_end = min(embargo_start + embargo_size, n_samples)

        embargo_indices = np.arange(embargo_start, embargo_end)

        # Remove embargo samples from training
        mask = ~np.isin(train_indices, embargo_indices)

        return train_indices[mask]

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits (sklearn compatibility)"""
        return self.n_splits


class CombinatorialPurgedKFold:
    """
    Combinatorial Purged K-Fold CV

    Generates multiple training/test split combinations while maintaining
    purging and embargo rules.Useful for more robust model evaluation.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,
        purge_gap: int = 0,
    ):
        """
        Initialize Combinatorial Purged K-Fold

        Args:
            n_splits: Number of splits
            n_test_groups: Number of groups to use as test set
            embargo_pct: Embargo percentage
            purge_gap: Purge gap size
        """
        self.n_splits = (n_splits,)
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct
        self.purge_gap = purge_gap

        # Calculate number of combinations
        from math import comb

        self.n_combinations = comb(n_splits, n_test_groups)

        logger.info(
            f"Initialized CombinatorialPurgedKFold: "
            f"{n_splits} splits, {n_test_groups} test groups, "
            f"{self.n_combinations} combinations"
        )

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        sample_times: pd.Series | None = None,
        pred_times: pd.Series | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate all combinations of train/test splits

        Args:
            X: Features array
            y: Target array
            sample_times: Sample start times
            pred_times: Prediction end times

        Yields:
            Tuple of (train_indices, test_indices)
        """
        from itertools import combinations

        n_samples = len(X)
        indices = np.arange(n_samples)

        # Divide into groups
        group_size = n_samples // self.n_splits
        groups = []

        for i in range(self.n_splits):
            start = i * group_size
            end = start + group_size if i < self.n_splits - 1 else n_samples
            groups.append(indices[start:end])

        # Generate all combinations of test groups
        for test_group_indices in combinations(range(self.n_splits), self.n_test_groups):
            # Combine selected groups as test set
            test_indices = np.concatenate([groups[i] for i in test_group_indices])

            # Rest are training candidates
            train_group_indices = [i for i in range(self.n_splits) if i not in test_group_indices]
            train_candidates = np.concatenate([groups[i] for i in train_group_indices])

            if sample_times is not None and pred_times is not None:
                # Apply purging and embargo
                test_times = sample_times.iloc[test_indices]
                test_start_time = test_times.min()
                test_end_time = pred_times.iloc[test_indices].max()

                # Purge overlapping samples
                train_indices = self._purge_overlapping(
                    train_candidates,
                    test_indices,
                    sample_times,
                    pred_times,
                    test_start_time,
                    test_end_time,
                )

                # Apply embargo
                train_indices = self._apply_embargo(
                    train_indices,
                    test_indices,
                    n_samples,
                )
            else:
                train_indices = train_candidates

            yield train_indices, test_indices

    def _purge_overlapping(
        self,
        train_candidates: np.ndarray,
        test_indices: np.ndarray,
        sample_times: pd.Series,
        pred_times: pd.Series,
        test_start_time: pd.Timestamp,
        test_end_time: pd.Timestamp,
    ) -> np.ndarray:
        """Purge overlapping samples (same logic as PurgedKFold)"""
        overlap_mask = np.ones(len(train_candidates), dtype=bool)

        for i, idx in enumerate(train_candidates):
            pred_time = pred_times.iloc[idx]

            if test_start_time <= pred_time <= test_end_time:
                overlap_mask[i] = False

        return train_candidates[overlap_mask]

    def _apply_embargo(
        self,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Apply embargo (same logic as PurgedKFold)"""
        if self.embargo_pct <= 0:
            return train_indices

        embargo_size = int(n_samples * self.embargo_pct)
        test_end = test_indices.max()
        embargo_start = test_end + 1
        embargo_end = min(embargo_start + embargo_size, n_samples)

        embargo_indices = np.arange(embargo_start, embargo_end)
        mask = ~np.isin(train_indices, embargo_indices)

        return train_indices[mask]

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of combinations"""
        return self.n_combinations


def create_sample_times(
    dates: pd.DatetimeIndex,
    horizon: int = 1,
) -> tuple[pd.Series, pd.Series]:
    """
    Create sample_times and pred_times from a date index

    Args:
        dates: DatetimeIndex of sample dates
        horizon: Prediction horizon in periods

    Returns:
        Tuple of (sample_times, pred_times)
    """
    sample_times = pd.Series(dates, index=range(len(dates)))

    # Prediction time is sample time + horizon
    pred_times = pd.Series(
        [dates[min(i + horizon, len(dates) - 1)] for i in range(len(dates))],
        index=range(len(dates)),
    )

    return sample_times, pred_times


def calculate_sample_weights(
    sample_times: pd.Series,
    pred_times: pd.Series,
    decay: float = 1.0,
) -> np.ndarray:
    """
    Calculate sample weights based on label uniqueness

    Samples with overlapping labels get lower weights to avoid overfitting
    to the same market regime.

    Args:
        sample_times: Sample start times
        pred_times: Prediction end times
        decay: Decay factor for time-based weighting

    Returns:
        Array of sample weights
    """
    n_samples = len(sample_times)
    weights = np.ones(n_samples)

    # For each sample, count how many other samples have overlapping labels
    for i in range(n_samples):
        sample_start = sample_times.iloc[i]
        sample_end = pred_times.iloc[i]

        # Count overlaps
        overlaps = 0
        for j in range(n_samples):
            if i == j:
                continue

            other_start = sample_times.iloc[j]
            other_end = pred_times.iloc[j]

            # Check if intervals overlap
            if not (sample_end < other_start or other_end < sample_start):
                overlaps += 1

        # Weight inversely proportional to overlaps
        if overlaps > 0:
            weights[i] = 1.0 / (1.0 + overlaps)

    # Apply time decay (more recent samples get higher weight)
    if decay != 1.0:
        time_weights = np.exp(-decay * np.arange(n_samples)[::-1])
        weights *= time_weights

    # Normalize
    weights /= weights.sum()

    return weights
