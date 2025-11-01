#!/usr/bin/env python3
"""
Inference Strategy Learning - Issue #23
Meta-learning system that learns which inference strategies work best

Phase 4: Self-Improvement - Meta-learning module
Tracks success/failure of different inference approaches and optimizes strategy selection
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import defaultdict


class InferenceStrategy(Enum):
    """Different inference strategies the system can use"""
    FORWARD_CHAINING = "forward_chaining"       # Start from facts, derive conclusions
    BACKWARD_CHAINING = "backward_chaining"     # Start from goal, find supporting facts
    ABDUCTIVE = "abductive"                     # Find best explanation
    ANALOGICAL = "analogical"                   # Reason by analogy
    DEDUCTIVE = "deductive"                     # Logical deduction
    INDUCTIVE = "inductive"                     # Generalize from examples
    TRANSITIVE = "transitive"                   # Follow transitive relations
    COMPOSITIONAL = "compositional"             # Compose relations
    HIERARCHICAL = "hierarchical"               # Navigate hierarchies (is-a)
    SIMILARITY_BASED = "similarity_based"       # Use structural similarity


class QueryType(Enum):
    """Types of queries to match strategies to"""
    CLASSIFICATION = "classification"           # "Is X a Y?"
    PROPERTY_QUERY = "property_query"          # "What properties does X have?"
    RELATION_QUERY = "relation_query"          # "What is the relation between X and Y?"
    PATH_FINDING = "path_finding"              # "How is X related to Y?"
    ANALOGY = "analogy"                        # "What is to X as Y is to Z?"
    GENERALIZATION = "generalization"          # "What do X, Y, Z have in common?"
    EXPLANATION = "explanation"                # "Why is X true?"
    PREDICTION = "prediction"                  # "What follows from X?"


@dataclass
class StrategyOutcome:
    """
    Record of applying an inference strategy
    Used for learning which strategies work
    """
    strategy: InferenceStrategy
    query_type: QueryType
    query: str
    success: bool
    execution_time: float
    confidence: float
    timestamp: float = field(default_factory=time.time)
    
    # Context for analysis
    result_quality: float = 0.0  # User feedback or automatic evaluation
    concepts_involved: Set[str] = field(default_factory=set)
    path_length: int = 0  # For path-finding queries
    
    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILURE"
        return (
            f"{status}: {self.strategy.value} on {self.query_type.value}\n"
            f"  Query: {self.query}\n"
            f"  Time: {self.execution_time:.3f}s, Confidence: {self.confidence:.2%}\n"
            f"  Quality: {self.result_quality:.2%}"
        )


@dataclass
class StrategyStats:
    """
    Statistics for a (strategy, query_type) pair
    """
    strategy: InferenceStrategy
    query_type: QueryType
    
    # Performance metrics
    success_count: int = 0
    failure_count: int = 0
    total_time: float = 0.0
    total_confidence: float = 0.0
    total_quality: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Percentage of successful applications"""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    @property
    def avg_time(self) -> float:
        """Average execution time"""
        total = self.success_count + self.failure_count
        return self.total_time / total if total > 0 else 0.0
    
    @property
    def avg_confidence(self) -> float:
        """Average confidence of results"""
        return self.total_confidence / self.success_count if self.success_count > 0 else 0.0
    
    @property
    def avg_quality(self) -> float:
        """Average result quality"""
        return self.total_quality / self.success_count if self.success_count > 0 else 0.0
    
    @property
    def score(self) -> float:
        """
        Overall score for this (strategy, query_type) combination
        Weighted combination of success rate, confidence, quality, and speed
        """
        if self.success_count == 0:
            return 0.0
        
        # Weights for different factors
        success_weight = 0.40
        quality_weight = 0.30
        confidence_weight = 0.20
        speed_weight = 0.10
        
        # Normalize speed (faster is better, cap at 10s)
        speed_score = max(0, 1.0 - (self.avg_time / 10.0))
        
        score = (
            success_weight * self.success_rate +
            quality_weight * self.avg_quality +
            confidence_weight * self.avg_confidence +
            speed_weight * speed_score
        )
        
        return score
    
    def update(self, outcome: StrategyOutcome):
        """Update statistics with a new outcome"""
        if outcome.success:
            self.success_count += 1
            self.total_confidence += outcome.confidence
            self.total_quality += outcome.result_quality
        else:
            self.failure_count += 1
        
        self.total_time += outcome.execution_time


class InferenceStrategyLearner:
    """
    Meta-learning system for inference strategies (Issue #23)
    
    Learns which strategies work best for different query types:
    - Tracks success/failure rates
    - Measures performance metrics
    - Recommends optimal strategies
    - Adapts over time
    """
    
    def __init__(
        self,
        min_samples: int = 5,
        exploration_rate: float = 0.2
    ):
        """
        Args:
            min_samples: Minimum samples before recommending a strategy
            exploration_rate: Probability of trying non-optimal strategy (exploration vs exploitation)
        """
        self.min_samples = min_samples
        self.exploration_rate = exploration_rate
        
        # Learning data structures
        self.outcomes: List[StrategyOutcome] = []
        self.stats: Dict[Tuple[InferenceStrategy, QueryType], StrategyStats] = {}
        
        # Strategy preferences per query type
        self.preferences: Dict[QueryType, List[InferenceStrategy]] = self._initialize_preferences()
    
    def _initialize_preferences(self) -> Dict[QueryType, List[InferenceStrategy]]:
        """
        Initialize default strategy preferences for each query type
        Based on theoretical suitability (refined through learning)
        """
        return {
            QueryType.CLASSIFICATION: [
                InferenceStrategy.HIERARCHICAL,
                InferenceStrategy.DEDUCTIVE,
                InferenceStrategy.SIMILARITY_BASED
            ],
            QueryType.PROPERTY_QUERY: [
                InferenceStrategy.FORWARD_CHAINING,
                InferenceStrategy.DEDUCTIVE,
                InferenceStrategy.COMPOSITIONAL
            ],
            QueryType.RELATION_QUERY: [
                InferenceStrategy.FORWARD_CHAINING,
                InferenceStrategy.BACKWARD_CHAINING,
                InferenceStrategy.TRANSITIVE
            ],
            QueryType.PATH_FINDING: [
                InferenceStrategy.BACKWARD_CHAINING,
                InferenceStrategy.TRANSITIVE,
                InferenceStrategy.HIERARCHICAL
            ],
            QueryType.ANALOGY: [
                InferenceStrategy.ANALOGICAL,
                InferenceStrategy.SIMILARITY_BASED,
                InferenceStrategy.ABDUCTIVE
            ],
            QueryType.GENERALIZATION: [
                InferenceStrategy.INDUCTIVE,
                InferenceStrategy.ABDUCTIVE,
                InferenceStrategy.SIMILARITY_BASED
            ],
            QueryType.EXPLANATION: [
                InferenceStrategy.ABDUCTIVE,
                InferenceStrategy.BACKWARD_CHAINING,
                InferenceStrategy.DEDUCTIVE
            ],
            QueryType.PREDICTION: [
                InferenceStrategy.FORWARD_CHAINING,
                InferenceStrategy.INDUCTIVE,
                InferenceStrategy.ANALOGICAL
            ]
        }
    
    def recommend_strategy(
        self,
        query_type: QueryType,
        concepts: Optional[Set[str]] = None,
        allow_exploration: bool = True
    ) -> InferenceStrategy:
        """
        Recommend best inference strategy for a query type
        
        Args:
            query_type: Type of query
            concepts: Concepts involved (for context-specific recommendations)
            allow_exploration: Whether to explore non-optimal strategies
            
        Returns:
            Recommended InferenceStrategy
        """
        import random
        
        # Exploration: occasionally try non-optimal strategies
        if allow_exploration and random.random() < self.exploration_rate:
            return random.choice(list(InferenceStrategy))
        
        # Get learned statistics for this query type
        candidates = []
        for strategy in InferenceStrategy:
            key = (strategy, query_type)
            if key in self.stats:
                stats = self.stats[key]
                # Only consider if we have enough samples
                if stats.success_count + stats.failure_count >= self.min_samples:
                    candidates.append((strategy, stats.score))
        
        # If we have learned preferences, use them
        if candidates:
            # Sort by score (descending)
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        # Fall back to default preferences
        if query_type in self.preferences:
            return self.preferences[query_type][0]
        
        # Ultimate fallback
        return InferenceStrategy.FORWARD_CHAINING
    
    def recommend_strategies(
        self,
        query_type: QueryType,
        n: int = 3
    ) -> List[Tuple[InferenceStrategy, float]]:
        """
        Recommend top N strategies with their scores
        
        Args:
            query_type: Type of query
            n: Number of strategies to recommend
            
        Returns:
            List of (strategy, score) tuples, sorted by score
        """
        candidates = []
        
        for strategy in InferenceStrategy:
            key = (strategy, query_type)
            if key in self.stats:
                stats = self.stats[key]
                # Include if we have at least one sample
                if stats.success_count + stats.failure_count > 0:
                    candidates.append((strategy, stats.score))
            else:
                # Use default preference order as weak signal
                if query_type in self.preferences:
                    prefs = self.preferences[query_type]
                    if strategy in prefs:
                        # Give small score based on default preference order
                        score = 0.1 / (prefs.index(strategy) + 1)
                        candidates.append((strategy, score))
        
        # Sort by score (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:n]
    
    def record_outcome(self, outcome: StrategyOutcome):
        """
        Record the outcome of applying an inference strategy
        Used for learning and adaptation
        
        Args:
            outcome: StrategyOutcome describing what happened
        """
        self.outcomes.append(outcome)
        
        # Update statistics
        key = (outcome.strategy, outcome.query_type)
        if key not in self.stats:
            self.stats[key] = StrategyStats(
                strategy=outcome.strategy,
                query_type=outcome.query_type
            )
        
        self.stats[key].update(outcome)
    
    def get_statistics(self, query_type: Optional[QueryType] = None) -> Dict[str, Any]:
        """
        Get learning statistics
        
        Args:
            query_type: Specific query type to analyze (None for all)
            
        Returns:
            Dictionary with statistics and insights
        """
        if query_type:
            # Statistics for specific query type
            relevant_stats = [
                stats for (strat, qtype), stats in self.stats.items()
                if qtype == query_type
            ]
        else:
            # All statistics
            relevant_stats = list(self.stats.values())
        
        if not relevant_stats:
            return {
                'total_outcomes': len(self.outcomes),
                'strategies_evaluated': 0,
                'message': 'Not enough data yet'
            }
        
        # Aggregate statistics
        total_success = sum(s.success_count for s in relevant_stats)
        total_failure = sum(s.failure_count for s in relevant_stats)
        total = total_success + total_failure
        
        # Find best strategies
        best_strategies = sorted(
            relevant_stats,
            key=lambda s: s.score,
            reverse=True
        )[:5]
        
        return {
            'total_outcomes': len(self.outcomes),
            'total_successes': total_success,
            'total_failures': total_failure,
            'overall_success_rate': total_success / total if total > 0 else 0.0,
            'strategies_evaluated': len(relevant_stats),
            'best_strategies': [
                {
                    'strategy': s.strategy.value,
                    'query_type': s.query_type.value,
                    'score': s.score,
                    'success_rate': s.success_rate,
                    'avg_confidence': s.avg_confidence,
                    'avg_quality': s.avg_quality,
                    'samples': s.success_count + s.failure_count
                }
                for s in best_strategies
            ]
        }
    
    def analyze_strategy(
        self,
        strategy: InferenceStrategy,
        query_type: Optional[QueryType] = None
    ) -> Dict[str, Any]:
        """
        Analyze performance of a specific strategy
        
        Args:
            strategy: Strategy to analyze
            query_type: Optional specific query type
            
        Returns:
            Analysis results
        """
        if query_type:
            key = (strategy, query_type)
            if key not in self.stats:
                return {
                    'strategy': strategy.value,
                    'query_type': query_type.value,
                    'message': 'No data for this combination'
                }
            
            stats = self.stats[key]
            return {
                'strategy': strategy.value,
                'query_type': query_type.value,
                'success_rate': stats.success_rate,
                'avg_time': stats.avg_time,
                'avg_confidence': stats.avg_confidence,
                'avg_quality': stats.avg_quality,
                'score': stats.score,
                'samples': stats.success_count + stats.failure_count
            }
        else:
            # Analyze across all query types
            relevant_stats = [
                stats for (strat, _), stats in self.stats.items()
                if strat == strategy
            ]
            
            if not relevant_stats:
                return {
                    'strategy': strategy.value,
                    'message': 'No data for this strategy'
                }
            
            total_success = sum(s.success_count for s in relevant_stats)
            total_failure = sum(s.failure_count for s in relevant_stats)
            total = total_success + total_failure
            
            return {
                'strategy': strategy.value,
                'total_samples': total,
                'success_rate': total_success / total if total > 0 else 0.0,
                'query_types': [
                    {
                        'query_type': s.query_type.value,
                        'score': s.score,
                        'samples': s.success_count + s.failure_count
                    }
                    for s in sorted(relevant_stats, key=lambda s: s.score, reverse=True)
                ]
            }
    
    def optimize_preferences(self):
        """
        Update default preferences based on learned statistics
        Called periodically to refine the system
        """
        for query_type in QueryType:
            # Get all strategies for this query type, sorted by score
            candidates = []
            for strategy in InferenceStrategy:
                key = (strategy, query_type)
                if key in self.stats:
                    stats = self.stats[key]
                    if stats.success_count + stats.failure_count >= self.min_samples:
                        candidates.append((strategy, stats.score))
            
            if candidates:
                # Sort by score (descending)
                candidates.sort(key=lambda x: x[1], reverse=True)
                # Update preferences with top strategies
                self.preferences[query_type] = [strat for strat, _ in candidates]
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """
        Get high-level insights about what the system has learned
        
        Returns:
            Human-readable insights
        """
        insights = {
            'total_experience': len(self.outcomes),
            'strategies_explored': len(set(o.strategy for o in self.outcomes)),
            'query_types_seen': len(set(o.query_type for o in self.outcomes)),
            'key_learnings': []
        }
        
        # Find best strategy for each query type
        for query_type in QueryType:
            best = None
            best_score = 0.0
            
            for strategy in InferenceStrategy:
                key = (strategy, query_type)
                if key in self.stats:
                    stats = self.stats[key]
                    if stats.score > best_score and stats.success_count >= self.min_samples:
                        best = strategy
                        best_score = stats.score
            
            if best:
                insights['key_learnings'].append({
                    'query_type': query_type.value,
                    'best_strategy': best.value,
                    'score': best_score
                })
        
        return insights


def demo():
    """Demonstrate inference strategy learning"""
    print("=" * 70)
    print("INFERENCE STRATEGY LEARNING - Issue #23 Demo")
    print("Meta-learning for optimal inference strategy selection")
    print("=" * 70)
    
    learner = InferenceStrategyLearner(min_samples=3, exploration_rate=0.1)
    
    # Simulate some outcomes
    print("\n1. SIMULATING LEARNING EXPERIENCE")
    print("-" * 70)
    
    outcomes = [
        # Classification queries work well with hierarchical reasoning
        StrategyOutcome(
            strategy=InferenceStrategy.HIERARCHICAL,
            query_type=QueryType.CLASSIFICATION,
            query="Is a dog a mammal?",
            success=True,
            execution_time=0.1,
            confidence=0.95,
            result_quality=0.9
        ),
        StrategyOutcome(
            strategy=InferenceStrategy.HIERARCHICAL,
            query_type=QueryType.CLASSIFICATION,
            query="Is a cat a mammal?",
            success=True,
            execution_time=0.08,
            confidence=0.93,
            result_quality=0.92
        ),
        StrategyOutcome(
            strategy=InferenceStrategy.DEDUCTIVE,
            query_type=QueryType.CLASSIFICATION,
            query="Is a bird a mammal?",
            success=True,
            execution_time=0.15,
            confidence=0.88,
            result_quality=0.85
        ),
        
        # Analogy queries work best with analogical reasoning
        StrategyOutcome(
            strategy=InferenceStrategy.ANALOGICAL,
            query_type=QueryType.ANALOGY,
            query="Hand is to human as paw is to?",
            success=True,
            execution_time=0.3,
            confidence=0.85,
            result_quality=0.88
        ),
        StrategyOutcome(
            strategy=InferenceStrategy.ANALOGICAL,
            query_type=QueryType.ANALOGY,
            query="Wheel is to car as wing is to?",
            success=True,
            execution_time=0.28,
            confidence=0.82,
            result_quality=0.86
        ),
        StrategyOutcome(
            strategy=InferenceStrategy.SIMILARITY_BASED,
            query_type=QueryType.ANALOGY,
            query="King is to queen as prince is to?",
            success=True,
            execution_time=0.35,
            confidence=0.75,
            result_quality=0.70
        ),
        
        # Forward chaining doesn't work well for analogies
        StrategyOutcome(
            strategy=InferenceStrategy.FORWARD_CHAINING,
            query_type=QueryType.ANALOGY,
            query="Sun is to day as moon is to?",
            success=False,
            execution_time=0.5,
            confidence=0.3,
            result_quality=0.2
        ),
    ]
    
    for outcome in outcomes:
        learner.record_outcome(outcome)
        print(f"  Recorded: {outcome.strategy.value} on {outcome.query_type.value} -> {'✓' if outcome.success else '✗'}")
    
    # Show what it learned
    print("\n2. LEARNED STRATEGY RECOMMENDATIONS")
    print("-" * 70)
    
    for query_type in [QueryType.CLASSIFICATION, QueryType.ANALOGY]:
        print(f"\n{query_type.value.upper()}:")
        recommendations = learner.recommend_strategies(query_type, n=3)
        for i, (strategy, score) in enumerate(recommendations, 1):
            print(f"  {i}. {strategy.value:20s} (score: {score:.3f})")
    
    # Analyze specific strategies
    print("\n3. STRATEGY ANALYSIS")
    print("-" * 70)
    
    for strategy in [InferenceStrategy.HIERARCHICAL, InferenceStrategy.ANALOGICAL]:
        print(f"\n{strategy.value.upper()}:")
        analysis = learner.analyze_strategy(strategy)
        if 'query_types' in analysis:
            print(f"  Total samples: {analysis['total_samples']}")
            print(f"  Overall success rate: {analysis['success_rate']:.2%}")
            print(f"  Best for:")
            for qt in analysis['query_types'][:3]:
                print(f"    - {qt['query_type']:15s} (score: {qt['score']:.3f}, n={qt['samples']})")
    
    # Show overall statistics
    print("\n4. LEARNING STATISTICS")
    print("-" * 70)
    
    stats = learner.get_statistics()
    print(f"  Total outcomes: {stats['total_outcomes']}")
    print(f"  Success rate: {stats['overall_success_rate']:.2%}")
    print(f"  Strategies evaluated: {stats['strategies_evaluated']}")
    print(f"\n  Best strategy-query combinations:")
    for i, best in enumerate(stats['best_strategies'][:3], 1):
        print(f"    {i}. {best['strategy']:20s} + {best['query_type']:15s}")
        print(f"       Score: {best['score']:.3f}, Success: {best['success_rate']:.2%}, n={best['samples']}")
    
    # Show insights
    print("\n5. KEY LEARNINGS")
    print("-" * 70)
    
    insights = learner.get_learning_insights()
    print(f"  Total experience: {insights['total_experience']} queries")
    print(f"  Strategies explored: {insights['strategies_explored']}")
    print(f"  Query types seen: {insights['query_types_seen']}")
    print(f"\n  Discovered best practices:")
    for learning in insights['key_learnings']:
        print(f"    • For {learning['query_type']:15s} → use {learning['best_strategy']:20s} (score: {learning['score']:.3f})")
    
    print("\n" + "=" * 70)
    print("KEY CAPABILITY: System learns which reasoning strategies work best!")
    print("=" * 70)


if __name__ == "__main__":
    demo()
