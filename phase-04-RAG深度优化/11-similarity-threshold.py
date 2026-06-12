#!/usr/bin/env python3
"""
Phase 04 Module 11: Similarity Threshold Optimization
========================================================
- Configurable per-collection thresholds: general 0.75, professional 0.85
- Below-threshold → discard before reranking
- Threshold tuning: grid search on validation set
- Recall vs precision trade-off visualization
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ===========================================================================
# Enums and Data Classes
# ===========================================================================

class Scenario(Enum):
    GENERAL = "general"         # 通用场景
    PROFESSIONAL = "professional"  # 专业场景


@dataclass
class ThresholdConfig:
    """Configuration for similarity thresholds per scenario/collection."""
    scenario: Scenario
    default_threshold: float = 0.75
    collection_thresholds: Dict[str, float] = field(default_factory=dict)
    min_threshold: float = 0.50
    max_threshold: float = 0.95


@dataclass
class ThresholdEvaluation:
    """Evaluation metrics for a specific threshold."""
    threshold: float
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    num_retained: int = 0
    num_discarded: int = 0
    total_candidates: int = 0


@dataclass
class GridSearchResult:
    """Result of a grid search over thresholds."""
    best_threshold: float
    best_f1: float
    best_precision: float
    best_recall: float
    all_evaluations: List[ThresholdEvaluation] = field(default_factory=list)
    scenario: Scenario = Scenario.GENERAL


# ===========================================================================
# Default Thresholds
# ===========================================================================

DEFAULT_THRESHOLDS: Dict[Scenario, float] = {
    Scenario.GENERAL: 0.75,
    Scenario.PROFESSIONAL: 0.85,
}

COLLECTION_SPECIFIC_THRESHOLDS: Dict[str, Dict[Scenario, float]] = {
    # Higher precision needed for these
    'legal_documents': {Scenario.GENERAL: 0.85, Scenario.PROFESSIONAL: 0.90},
    'medical_records': {Scenario.GENERAL: 0.85, Scenario.PROFESSIONAL: 0.90},
    'financial_reports': {Scenario.GENERAL: 0.82, Scenario.PROFESSIONAL: 0.88},
    # Slightly relaxed for these
    'faq_knowledge_base': {Scenario.GENERAL: 0.70, Scenario.PROFESSIONAL: 0.80},
    'technical_docs': {Scenario.GENERAL: 0.75, Scenario.PROFESSIONAL: 0.85},
    'tutorials': {Scenario.GENERAL: 0.72, Scenario.PROFESSIONAL: 0.82},
}


# ===========================================================================
# Threshold Manager
# ===========================================================================

class SimilarityThresholdManager:
    """Manage similarity thresholds with collection-specific overrides."""

    def __init__(
        self,
        scenario: Scenario = Scenario.GENERAL,
        custom_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.scenario = scenario
        self.default_threshold = DEFAULT_THRESHOLDS[scenario]

        # Build collection thresholds
        self.collection_thresholds: Dict[str, float] = {}
        for collection, thresholds in COLLECTION_SPECIFIC_THRESHOLDS.items():
            self.collection_thresholds[collection] = thresholds.get(
                scenario, self.default_threshold
            )

        # Apply custom overrides
        if custom_thresholds:
            self.collection_thresholds.update(custom_thresholds)

    def get_threshold(self, collection: Optional[str] = None) -> float:
        """Get the threshold for a specific collection or the default."""
        if collection and collection in self.collection_thresholds:
            return self.collection_thresholds[collection]
        return self.default_threshold

    def set_threshold(self, collection: str, threshold: float) -> None:
        """Set a custom threshold for a collection."""
        self.collection_thresholds[collection] = max(0.0, min(1.0, threshold))

    def filter(self, results: List[Tuple[str, float, Any]],
               collection: Optional[str] = None) -> List[Tuple[str, float, Any]]:
        """Filter results below the similarity threshold.

        Args:
            results: List of (doc_id, similarity_score, metadata).
            collection: Optional collection name for specific threshold.

        Returns:
            Filtered list with below-threshold items removed.
        """
        threshold = self.get_threshold(collection)
        return [(did, score, meta) for did, score, meta in results
                if score >= threshold]

    def get_discarded_count(self, results: List[Tuple[str, float]],
                           collection: Optional[str] = None) -> int:
        """Count how many results would be discarded."""
        threshold = self.get_threshold(collection)
        return sum(1 for _, score in results if score < threshold)

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as a dictionary."""
        return {
            'scenario': self.scenario.value,
            'default_threshold': self.default_threshold,
            'collection_thresholds': self.collection_thresholds,
        }


# ===========================================================================
# Grid Search Threshold Tuner
# ===========================================================================

class ThresholdTuner:
    """Tune similarity threshold via grid search on validation data."""

    def __init__(
        self,
        scenario: Scenario = Scenario.GENERAL,
        min_threshold: float = 0.50,
        max_threshold: float = 0.95,
        step: float = 0.05,
    ):
        self.scenario = scenario
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.step = step

    def grid_search(
        self,
        validation_queries: List[str],
        validation_results: List[List[Tuple[str, float]]],
        relevance_labels: List[Set[str]],
        metric: str = 'f1',
    ) -> GridSearchResult:
        """Perform grid search to find the optimal threshold.

        Args:
            validation_queries: List of test queries.
            validation_results: For each query, list of (doc_id, score).
            relevance_labels: For each query, set of relevant doc_ids.
            metric: Optimization metric ('f1', 'precision', 'recall').

        Returns:
            GridSearchResult with best threshold and full evaluation.
        """
        thresholds = np.arange(self.min_threshold, self.max_threshold + self.step,
                              self.step)
        evaluations: List[ThresholdEvaluation] = []

        best_eval: Optional[ThresholdEvaluation] = None
        best_metric_value = -1.0

        for threshold in thresholds:
            ev = self._evaluate_threshold(
                threshold, validation_results, relevance_labels
            )
            evaluations.append(ev)

            metric_value = getattr(ev, metric, ev.f1_score)
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_eval = ev

        if best_eval is None:
            return GridSearchResult(
                best_threshold=self.min_threshold,
                best_f1=0.0,
                best_precision=0.0,
                best_recall=0.0,
                all_evaluations=evaluations,
                scenario=self.scenario,
            )

        return GridSearchResult(
            best_threshold=best_eval.threshold,
            best_f1=best_eval.f1_score,
            best_precision=best_eval.precision,
            best_recall=best_eval.recall,
            all_evaluations=evaluations,
            scenario=self.scenario,
        )

    def _evaluate_threshold(
        self,
        threshold: float,
        all_results: List[List[Tuple[str, float]]],
        relevance_labels: List[Set[str]],
    ) -> ThresholdEvaluation:
        """Evaluate metrics at a specific threshold."""
        ev = ThresholdEvaluation(threshold=threshold)

        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_retained = 0
        total_discarded = 0
        total_candidates = 0

        for results, relevant in zip(all_results, relevance_labels):
            # Filter by threshold
            filtered = [(did, score) for did, score in results if score >= threshold]
            retained_ids = {did for did, _ in filtered}
            relevant_set = set(relevant)

            total_retained += len(retained_ids)
            total_discarded += len(results) - len(retained_ids)
            total_candidates += len(results)

            # Precision
            if retained_ids:
                tp = len(retained_ids & relevant_set)
                precision = tp / len(retained_ids)
            else:
                precision = 0.0

            # Recall
            if relevant_set:
                recall = len(retained_ids & relevant_set) / len(relevant_set)
            else:
                recall = 1.0

            # F1
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            total_precision += precision
            total_recall += recall
            total_f1 += f1

        n = len(all_results)
        if n > 0:
            ev.precision = total_precision / n
            ev.recall = total_recall / n
            ev.f1_score = total_f1 / n
        ev.num_retained = total_retained
        ev.num_discarded = total_discarded
        ev.total_candidates = total_candidates

        return ev

    def tune_and_visualize(
        self,
        validation_queries: List[str],
        validation_results: List[List[Tuple[str, float]]],
        relevance_labels: List[Set[str]],
        output_path: Optional[str] = None,
    ) -> GridSearchResult:
        """Tune threshold and generate a visualization."""
        result = self.grid_search(
            validation_queries, validation_results, relevance_labels
        )

        if HAS_MATPLOTLIB:
            self._plot_tradeoff(result, output_path)

        return result

    def _plot_tradeoff(
        self,
        result: GridSearchResult,
        output_path: Optional[str] = None,
    ) -> None:
        """Plot precision-recall trade-off across thresholds."""
        if not result.all_evaluations:
            return

        thresholds = [e.threshold for e in result.all_evaluations]
        precisions = [e.precision for e in result.all_evaluations]
        recalls = [e.recall for e in result.all_evaluations]
        f1_scores = [e.f1_score for e in result.all_evaluations]
        num_retained = [e.num_retained for e in result.all_evaluations]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Precision-Recall Curve
        ax1 = axes[0]
        ax1.plot(thresholds, precisions, 'b-o', label='Precision', linewidth=2,
                markersize=6)
        ax1.plot(thresholds, recalls, 'r-s', label='Recall', linewidth=2,
                markersize=6)
        ax1.plot(thresholds, f1_scores, 'g-^', label='F1 Score', linewidth=2,
                markersize=6)
        best_idx = thresholds.index(result.best_threshold) if result.best_threshold in thresholds else -1
        if best_idx >= 0:
            ax1.axvline(x=result.best_threshold, color='purple', linestyle='--',
                       alpha=0.7, label=f'Best={result.best_threshold:.2f}')
        ax1.set_xlabel('Similarity Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title(f'Precision-Recall vs Threshold\nScenario: {result.scenario.value}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)

        # Plot 2: F1 Score focus
        ax2 = axes[1]
        ax2.fill_between(thresholds, 0, f1_scores, alpha=0.3, color='green')
        ax2.plot(thresholds, f1_scores, 'g-', linewidth=3)
        if best_idx >= 0:
            ax2.plot(result.best_threshold, result.best_f1, 'r*', markersize=15,
                    label=f'Max F1={result.best_f1:.3f} @ {result.best_threshold:.2f}')
        ax2.set_xlabel('Similarity Threshold')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score Optimization')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)

        # Plot 3: Results retained
        ax3 = axes[2]
        ax3.bar(thresholds, num_retained, width=0.03, color='steelblue',
               edgecolor='white')
        ax3.set_xlabel('Similarity Threshold')
        ax3.set_ylabel('Documents Retained')
        ax3.set_title('Retention vs Threshold')
        ax3.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")
        else:
            plt.show()

        plt.close()


# ===========================================================================
# Threshold Application in Retrieval Pipeline
# ===========================================================================

class ThresholdFilter:
    """Apply similarity thresholds in a retrieval pipeline.

    Usage in a pipeline:
    1. Coarse retrieval returns Top-N candidates with scores
    2. ThresholdFilter discards below-threshold items
    3. Remaining items go to reranker
    4. This reduces reranker workload and improves quality
    """

    def __init__(
        self,
        threshold_manager: SimilarityThresholdManager,
    ):
        self.manager = threshold_manager

    def apply(
        self,
        candidates: List[Tuple[str, float, Any]],
        collection: Optional[str] = None,
    ) -> Tuple[List[Tuple[str, float, Any]], int]:
        """Apply threshold filtering.

        Returns:
            (filtered_candidates, num_discarded).
        """
        before = len(candidates)
        filtered = self.manager.filter(candidates, collection)
        after = len(filtered)
        discarded = before - after

        return filtered, discarded

    def should_discard(self, score: float, collection: Optional[str] = None) -> bool:
        """Check if a single result should be discarded."""
        threshold = self.manager.get_threshold(collection)
        return score < threshold


# ===========================================================================
# __main__: Demo
# ===========================================================================

def _generate_synthetic_data(
    n_queries: int = 10,
    n_docs_per_query: int = 20,
    n_relevant_per_query: int = 3,
    seed: int = 42,
) -> Tuple[List[str], List[List[Tuple[str, float]]], List[Set[str]]]:
    """Generate synthetic validation data for threshold tuning."""
    rng = np.random.RandomState(seed)

    queries = [f"query_{i:02d}" for i in range(n_queries)]
    all_results: List[List[Tuple[str, float]]] = []
    all_relevant: List[Set[str]] = []

    for qi in range(n_queries):
        relevant_docs = {f"doc_{qi}_{rj}" for rj in range(n_relevant_per_query)}
        # Generate scores: relevant docs get higher scores with noise
        results: List[Tuple[str, float]] = []

        for di in range(n_docs_per_query):
            doc_id = f"doc_{qi}_{di}"
            if doc_id in relevant_docs:
                # Relevant: high score with noise
                score = 0.75 + rng.random() * 0.25  # 0.75-1.0
            else:
                # Irrelevant: varied score
                score = rng.random() * 0.85  # 0.0-0.85
            results.append((doc_id, float(score)))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        all_results.append(results)
        all_relevant.append(relevant_docs)

    return queries, all_results, all_relevant


if __name__ == '__main__':
    print("=" * 60)
    print("Similarity Threshold Optimization - Self-Test")
    print("=" * 60)

    # --- Test 1: Threshold Manager ---
    print("\n[Test 1] SimilarityThresholdManager")
    print("-" * 40)

    for scenario in [Scenario.GENERAL, Scenario.PROFESSIONAL]:
        mgr = SimilarityThresholdManager(scenario=scenario)
        print(f"\n  Scenario: {scenario.value}")
        print(f"  Default threshold:            {mgr.default_threshold:.2f}")
        print(f"  legal_documents threshold:     {mgr.get_threshold('legal_documents'):.2f}")
        print(f"  faq_knowledge_base threshold:  {mgr.get_threshold('faq_knowledge_base'):.2f}")
        print(f"  Unknown collection (default):  {mgr.get_threshold('unknown_coll'):.2f}")

        # Test filtering
        sample_results = [
            ("doc_a", 0.90, {}),
            ("doc_b", 0.82, {}),
            ("doc_c", 0.75, {}),
            ("doc_d", 0.70, {}),
            ("doc_e", 0.60, {}),
            ("doc_f", 0.45, {}),
        ]
        filtered = mgr.filter(sample_results)
        discarded = mgr.get_discarded_count(
            [(did, score) for did, score, _ in sample_results]
        )
        print(f"\n  Sample filtering (threshold={mgr.default_threshold:.2f}):")
        print(f"    Input:  {len(sample_results)} results")
        print(f"    Kept:   {len(filtered)} results")
        print(f"    Discarded: {discarded} results")
        for did, score, _ in filtered:
            print(f"      {did}: {score:.2f}")
        for did, score, _ in sample_results:
            if score < mgr.default_threshold:
                print(f"      {did}: {score:.2f} [DISCARDED: below threshold]")

    # --- Test 2: Custom collection threshold ---
    print(f"\n[Test 2] Custom Collection Threshold")
    print("-" * 40)

    mgr = SimilarityThresholdManager(scenario=Scenario.GENERAL)
    mgr.set_threshold('my_custom_collection', 0.88)
    print(f"  Set 'my_custom_collection' threshold to: {mgr.get_threshold('my_custom_collection'):.2f}")

    # --- Test 3: Grid Search Threshold Tuning ---
    print(f"\n[Test 3] Grid Search Threshold Tuning")
    print("-" * 40)

    queries, all_results, all_relevant = _generate_synthetic_data(
        n_queries=20, n_docs_per_query=20, n_relevant_per_query=4, seed=42
    )

    tuner = ThresholdTuner(
        scenario=Scenario.GENERAL,
        min_threshold=0.50,
        max_threshold=0.95,
        step=0.05,
    )

    result = tuner.grid_search(
        queries,
        all_results,
        all_relevant,
        metric='f1',
    )

    print(f"\n  Grid Search Results ({result.scenario.value}):")
    print(f"  Best Threshold: {result.best_threshold:.2f}")
    print(f"  Best F1:        {result.best_f1:.4f}")
    print(f"  Best Precision: {result.best_precision:.4f}")
    print(f"  Best Recall:    {result.best_recall:.4f}")

    print(f"\n  Full Evaluation Table:")
    print(f"  {'Threshold':<12} {'Precision':<12} {'Recall':<12} "
          f"{'F1':<12} {'Retained':<10} {'Discarded':<10}")
    print(f"  {'-'*68}")
    for ev in result.all_evaluations:
        marker = ' <-- BEST' if ev.threshold == result.best_threshold else ''
        print(f"  {ev.threshold:<12.2f} {ev.precision:<12.4f} {ev.recall:<12.4f} "
              f"{ev.f1_score:<12.4f} {ev.num_retained:<10} {ev.num_discarded:<10}{marker}")

    # --- Test 4: Professional scenario ---
    print(f"\n[Test 4] Professional Scenario Tuning")
    print("-" * 40)

    tuner_pro = ThresholdTuner(
        scenario=Scenario.PROFESSIONAL,
        min_threshold=0.70,
        max_threshold=0.95,
        step=0.05,
    )

    # Generate professional data (higher baseline scores for relevant docs)
    _, all_results_pro, all_relevant_pro = _generate_synthetic_data(
        n_queries=20, n_docs_per_query=20, n_relevant_per_query=4, seed=99
    )

    result_pro = tuner_pro.grid_search(
        queries, all_results_pro, all_relevant_pro, metric='f1'
    )

    print(f"\n  Grid Search Results ({result_pro.scenario.value}):")
    print(f"  Best Threshold: {result_pro.best_threshold:.2f}")
    print(f"  Best F1:        {result_pro.best_f1:.4f}")
    print(f"  Best Precision: {result_pro.best_precision:.4f}")
    print(f"  Best Recall:    {result_pro.best_recall:.4f}")

    # Compare
    print(f"\n  Comparison:")
    print(f"    General (0.75 default) → Best: {result.best_threshold:.2f} "
          f"(F1={result.best_f1:.4f})")
    print(f"    Professional (0.85 default) → Best: {result_pro.best_threshold:.2f} "
          f"(F1={result_pro.best_f1:.4f})")

    # --- Test 5: Visualization ---
    if HAS_MATPLOTLIB:
        print(f"\n[Test 5] Generating Visualization...")
        output_dir = os.path.dirname(os.path.abspath(__file__))
        viz_path = os.path.join(output_dir, 'threshold_tradeoff.png')
        tuner.tune_and_visualize(
            queries, all_results, all_relevant, output_path=viz_path
        )

    # --- Test 6: ThresholdFilter in a pipeline ---
    print(f"\n[Test 6] ThresholdFilter Pipeline Integration")
    print("-" * 40)

    mgr_filter = SimilarityThresholdManager(scenario=Scenario.GENERAL)
    threshold_filter = ThresholdFilter(mgr_filter)

    # Simulate coarse retrieval output
    coarse_results = [
        ("doc_1", 0.92, {"source": "legal"}),
        ("doc_2", 0.85, {"source": "wiki"}),
        ("doc_3", 0.78, {"source": "manual"}),
        ("doc_4", 0.72, {"source": "faq"}),
        ("doc_5", 0.65, {"source": "blog"}),
        ("doc_6", 0.55, {"source": "forum"}),
        ("doc_7", 0.48, {"source": "social"}),
        ("doc_8", 0.35, {"source": "unknown"}),
    ]

    print(f"\n  Coarse retrieval: {len(coarse_results)} candidates")
    filtered, discarded = threshold_filter.apply(coarse_results)
    print(f"  After threshold ({mgr_filter.default_threshold:.2f}): "
          f"{len(filtered)} kept, {discarded} discarded")
    for did, score, meta in filtered:
        print(f"    KEPT: {did} ({score:.2f}) {meta}")
    for did, score, meta in coarse_results:
        if score < mgr_filter.default_threshold:
            print(f"    DISCARDED: {did} ({score:.2f}) {meta}")

    print(f"\n  → Reranker would process {len(filtered)} items instead of "
          f"{len(coarse_results)}, reducing cost by "
          f"{(1 - len(filtered)/len(coarse_results))*100:.0f}%")

    print(f"\n{'='*60}")
    print("Similarity Threshold self-test completed!")
    print(f"{'='*60}")
