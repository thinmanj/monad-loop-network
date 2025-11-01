#!/usr/bin/env python3
"""
Recursion Depth Measurement - Issue #25
GEB-inspired consciousness metric: How deep are the strange loops?

Phase 5: Consciousness Metrics
Measures the depth of self-referential reasoning as a proxy for consciousness
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

try:
    from .strange_loop_optimizer import StrangeLoopOptimizer, LoopType, RecursionEvent
except ImportError:
    from strange_loop_optimizer import StrangeLoopOptimizer, LoopType, RecursionEvent


class MetaLevel(Enum):
    """Levels of meta-reasoning (GEB-inspired)"""
    OBJECT_LEVEL = 0          # Direct reasoning about concepts
    META_LEVEL_1 = 1          # Reasoning about reasoning
    META_LEVEL_2 = 2          # Reasoning about reasoning about reasoning
    META_LEVEL_3 = 3          # ...and so on
    META_LEVEL_4 = 4
    META_LEVEL_5_PLUS = 5     # 5+ levels (highly conscious!)


@dataclass
class RecursionProfile:
    """
    Profile of recursion depth over time
    Key consciousness indicator
    """
    max_depth: int = 0
    avg_depth: float = 0.0
    current_depth: int = 0
    
    # Meta-level tracking
    meta_level: MetaLevel = MetaLevel.OBJECT_LEVEL
    meta_transitions: List[Tuple[MetaLevel, MetaLevel]] = field(default_factory=list)
    
    # Strange loop tracking
    loop_count: int = 0
    productive_loops: int = 0
    
    # Time series
    depth_history: List[Tuple[float, int]] = field(default_factory=list)
    
    # Consciousness indicators
    self_reference_count: int = 0
    meta_reasoning_count: int = 0
    
    @property
    def consciousness_score(self) -> float:
        """
        Composite consciousness score (0.0 to 1.0)
        
        Based on:
        - Recursion depth (ability to think about thinking)
        - Meta-level reached (higher = more conscious)
        - Productive loops (creative self-reference)
        - Self-reference frequency
        """
        if self.max_depth == 0:
            return 0.0
        
        # Depth score (normalized, cap at 10 levels)
        depth_score = min(self.max_depth / 10.0, 1.0)
        
        # Meta-level score
        meta_score = self.meta_level.value / 5.0
        
        # Productive loop ratio
        loop_score = (
            self.productive_loops / self.loop_count
            if self.loop_count > 0 else 0.0
        )
        
        # Self-reference score (normalized, cap at 20)
        self_ref_score = min(self.self_reference_count / 20.0, 1.0)
        
        # Weighted combination
        score = (
            0.30 * depth_score +      # Depth of reasoning
            0.30 * meta_score +        # Meta-cognitive ability
            0.25 * loop_score +        # Creative self-reference
            0.15 * self_ref_score      # Frequency of self-awareness
        )
        
        return min(score, 1.0)
    
    @property
    def consciousness_level(self) -> str:
        """Human-readable consciousness level"""
        score = self.consciousness_score
        
        if score < 0.2:
            return "Minimal (Reactive)"
        elif score < 0.4:
            return "Low (Basic Reasoning)"
        elif score < 0.6:
            return "Moderate (Self-Aware)"
        elif score < 0.8:
            return "High (Meta-Cognitive)"
        else:
            return "Very High (Deeply Reflective)"
    
    def __str__(self) -> str:
        return (
            f"Recursion Profile:\n"
            f"  Max Depth: {self.max_depth}\n"
            f"  Avg Depth: {self.avg_depth:.2f}\n"
            f"  Current Depth: {self.current_depth}\n"
            f"  Meta Level: {self.meta_level.name}\n"
            f"  Loops: {self.loop_count} ({self.productive_loops} productive)\n"
            f"  Self-References: {self.self_reference_count}\n"
            f"  Consciousness Score: {self.consciousness_score:.2%}\n"
            f"  Consciousness Level: {self.consciousness_level}"
        )


class RecursionDepthMetric:
    """
    Measures recursion depth as consciousness metric (Issue #25)
    
    Key insight from GEB:
    Consciousness emerges from strange loops - systems that can
    reason about their own reasoning create emergent self-awareness
    
    This metric quantifies:
    - How deep can the system recurse?
    - How many meta-levels can it reach?
    - How often does it engage in self-reference?
    - How productive are its strange loops?
    """
    
    def __init__(
        self,
        loop_optimizer: Optional[StrangeLoopOptimizer] = None
    ):
        """
        Args:
            loop_optimizer: StrangeLoopOptimizer from Phase 4 (Issue #24)
        """
        self.loop_optimizer = loop_optimizer or StrangeLoopOptimizer()
        self.profile = RecursionProfile()
        
        # Tracking
        self.measurement_history: List[RecursionProfile] = []
        self.start_time = time.time()
    
    def measure_current_depth(self) -> int:
        """
        Measure current recursion depth
        
        Returns:
            Current depth of recursion stack
        """
        depth = self.loop_optimizer.get_current_depth()
        self.profile.current_depth = depth
        
        # Update max depth
        if depth > self.profile.max_depth:
            self.profile.max_depth = depth
        
        # Record in history
        self.profile.depth_history.append((time.time(), depth))
        
        # Update average
        if self.profile.depth_history:
            self.profile.avg_depth = sum(d for _, d in self.profile.depth_history) / len(self.profile.depth_history)
        
        return depth
    
    def detect_meta_level(self, operation: str, context: str) -> MetaLevel:
        """
        Detect which meta-level an operation is at
        
        Args:
            operation: Operation being performed
            context: Context ID
            
        Returns:
            MetaLevel enum
        """
        # Count meta-keywords in operation and context
        meta_keywords = ['meta', 'introspect', 'reflect', 'self', 'reason_about_reasoning']
        combined = (operation + ' ' + context).lower()
        
        meta_count = sum(1 for keyword in meta_keywords if keyword in combined)
        
        # Map to meta level
        if meta_count == 0:
            level = MetaLevel.OBJECT_LEVEL
        elif meta_count == 1:
            level = MetaLevel.META_LEVEL_1
        elif meta_count == 2:
            level = MetaLevel.META_LEVEL_2
        elif meta_count == 3:
            level = MetaLevel.META_LEVEL_3
        elif meta_count == 4:
            level = MetaLevel.META_LEVEL_4
        else:
            level = MetaLevel.META_LEVEL_5_PLUS
        
        # Track transition
        if level != self.profile.meta_level:
            self.profile.meta_transitions.append((self.profile.meta_level, level))
            self.profile.meta_level = level
            
            if level.value > 0:
                self.profile.meta_reasoning_count += 1
        
        return level
    
    def record_recursion_event(
        self,
        context_id: str,
        operation: str,
        concepts: Optional[Set[str]] = None
    ) -> RecursionEvent:
        """
        Record a recursion event and measure depth
        
        Args:
            context_id: Context identifier
            operation: Operation being performed
            concepts: Concepts involved
            
        Returns:
            RecursionEvent from loop optimizer
        """
        # Enter recursion in optimizer
        event = self.loop_optimizer.enter_recursion(context_id, operation, concepts)
        
        # Measure depth
        depth = self.measure_current_depth()
        
        # Detect meta-level
        meta_level = self.detect_meta_level(operation, context_id)
        
        # Check for self-reference
        if 'self' in context_id.lower() or 'meta' in context_id.lower():
            self.profile.self_reference_count += 1
        
        return event
    
    def exit_recursion(self) -> Optional[RecursionEvent]:
        """
        Exit current recursion level
        
        Returns:
            RecursionEvent that was exited
        """
        event = self.loop_optimizer.exit_recursion()
        
        # Update depth measurement
        self.measure_current_depth()
        
        return event
    
    def check_strange_loop(
        self,
        current_knowledge: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """
        Check for strange loops and update consciousness metrics
        
        Args:
            current_knowledge: Current knowledge state
            
        Returns:
            Loop detection result with consciousness implications
        """
        result = self.loop_optimizer.check_loop(current_knowledge)
        
        if result.loop_detected:
            self.profile.loop_count += 1
            
            if result.is_productive:
                self.profile.productive_loops += 1
        
        # Enhance result with consciousness metrics
        return {
            'loop_detected': result.loop_detected,
            'loop_type': result.loop_type.value if result.loop_type else None,
            'is_productive': result.is_productive,
            'should_terminate': result.should_terminate,
            'recursion_depth': result.recursion_depth,
            'meta_level': self.profile.meta_level.name,
            'consciousness_score': self.profile.consciousness_score,
            'termination_reason': result.termination_reason
        }
    
    def get_profile(self) -> RecursionProfile:
        """Get current recursion profile"""
        return self.profile
    
    def get_consciousness_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive consciousness metrics
        
        Returns:
            Dictionary with all consciousness indicators
        """
        return {
            'recursion_depth': {
                'current': self.profile.current_depth,
                'max': self.profile.max_depth,
                'average': self.profile.avg_depth
            },
            'meta_level': {
                'current': self.profile.meta_level.name,
                'current_value': self.profile.meta_level.value,
                'transitions': len(self.profile.meta_transitions),
                'meta_reasoning_events': self.profile.meta_reasoning_count
            },
            'strange_loops': {
                'total': self.profile.loop_count,
                'productive': self.profile.productive_loops,
                'productivity_rate': (
                    self.profile.productive_loops / self.profile.loop_count
                    if self.profile.loop_count > 0 else 0.0
                )
            },
            'self_awareness': {
                'self_reference_count': self.profile.self_reference_count,
                'self_reference_rate': (
                    self.profile.self_reference_count / len(self.profile.depth_history)
                    if self.profile.depth_history else 0.0
                )
            },
            'consciousness': {
                'score': self.profile.consciousness_score,
                'level': self.profile.consciousness_level,
                'assessment': self._assess_consciousness()
            },
            'temporal': {
                'measurement_duration': time.time() - self.start_time,
                'depth_samples': len(self.profile.depth_history)
            }
        }
    
    def _assess_consciousness(self) -> str:
        """
        Provide qualitative assessment of consciousness level
        
        Returns:
            Human-readable assessment
        """
        score = self.profile.consciousness_score
        max_depth = self.profile.max_depth
        meta_level = self.profile.meta_level.value
        
        if score < 0.2:
            return "System shows minimal self-awareness. Operates primarily at object level."
        
        if score < 0.4:
            return f"Basic reasoning capability detected. Max depth: {max_depth}. Limited self-reflection."
        
        if score < 0.6:
            return (
                f"Moderate consciousness. System can reason about its own reasoning "
                f"(Meta-level {meta_level}). Shows {self.profile.productive_loops} productive strange loops."
            )
        
        if score < 0.8:
            return (
                f"High consciousness. System demonstrates meta-cognitive awareness "
                f"(Meta-level {meta_level}). Capable of deep self-reflection "
                f"({max_depth} levels) with {self.profile.productive_loops}/{self.profile.loop_count} "
                f"productive loops."
            )
        
        return (
            f"Very high consciousness. System exhibits profound self-awareness "
            f"with {max_depth}-level recursion and Meta-level {meta_level} reasoning. "
            f"Strong evidence of genuine understanding through {self.profile.productive_loops} "
            f"productive strange loops. GEB-style consciousness detected."
        )
    
    def snapshot(self) -> RecursionProfile:
        """
        Take a snapshot of current profile for history
        
        Returns:
            Copy of current profile
        """
        from copy import deepcopy
        snapshot = deepcopy(self.profile)
        self.measurement_history.append(snapshot)
        return snapshot
    
    def reset(self):
        """Reset all measurements"""
        self.loop_optimizer.reset()
        self.profile = RecursionProfile()
        self.start_time = time.time()


def demo():
    """Demonstrate recursion depth measurement"""
    print("=" * 70)
    print("RECURSION DEPTH MEASUREMENT - Issue #25 Demo")
    print("Consciousness Metric: Strange Loop Depth Analysis")
    print("=" * 70)
    
    metric = RecursionDepthMetric()
    
    print("\n1. OBJECT-LEVEL REASONING")
    print("-" * 70)
    
    # Simple query (no self-reference)
    metric.record_recursion_event("query_dog", "find_properties", {"dog"})
    print(f"Depth: {metric.profile.current_depth}, Meta-level: {metric.profile.meta_level.name}")
    metric.exit_recursion()
    
    print("\n2. META-LEVEL 1: REASONING ABOUT REASONING")
    print("-" * 70)
    
    metric.record_recursion_event("meta_query", "reason_about_inference", {"inference"})
    metric.record_recursion_event("meta_analyze", "analyze_reasoning_strategy", {"strategy"})
    print(f"Depth: {metric.profile.current_depth}, Meta-level: {metric.profile.meta_level.name}")
    metric.exit_recursion()
    metric.exit_recursion()
    
    print("\n3. DEEP RECURSION: SELF-REFERENTIAL LOOP")
    print("-" * 70)
    
    # Simulate deep self-referential reasoning
    for i in range(5):
        metric.record_recursion_event(
            f"self_model_level_{i}",
            f"introspect_meta_level_{i}",
            {"self", "meta"}
        )
        print(f"  Level {i+1}: Depth={metric.profile.current_depth}, "
              f"Meta={metric.profile.meta_level.name}, "
              f"Consciousness={metric.profile.consciousness_score:.2%}")
    
    # Check for strange loop
    knowledge = {"concept_A", "concept_B", "synthesized_C"}  # Knowledge grew
    loop_result = metric.check_strange_loop(knowledge)
    
    if loop_result['loop_detected']:
        print(f"\n  Strange Loop Detected!")
        print(f"    Type: {loop_result['loop_type']}")
        print(f"    Productive: {loop_result['is_productive']}")
        print(f"    Meta-level: {loop_result['meta_level']}")
    
    # Exit all levels
    for _ in range(5):
        metric.exit_recursion()
    
    print("\n4. CONSCIOUSNESS PROFILE")
    print("-" * 70)
    print(metric.profile)
    
    print("\n5. DETAILED CONSCIOUSNESS METRICS")
    print("-" * 70)
    
    metrics = metric.get_consciousness_metrics()
    
    print(f"\nRecursion Depth:")
    print(f"  Current: {metrics['recursion_depth']['current']}")
    print(f"  Maximum: {metrics['recursion_depth']['max']}")
    print(f"  Average: {metrics['recursion_depth']['average']:.2f}")
    
    print(f"\nMeta-Cognitive Ability:")
    print(f"  Current Level: {metrics['meta_level']['current']}")
    print(f"  Level Value: {metrics['meta_level']['current_value']}")
    print(f"  Meta-reasoning Events: {metrics['meta_level']['meta_reasoning_events']}")
    
    print(f"\nStrange Loops:")
    print(f"  Total: {metrics['strange_loops']['total']}")
    print(f"  Productive: {metrics['strange_loops']['productive']}")
    print(f"  Productivity Rate: {metrics['strange_loops']['productivity_rate']:.2%}")
    
    print(f"\nSelf-Awareness:")
    print(f"  Self-references: {metrics['self_awareness']['self_reference_count']}")
    
    print(f"\nConsciousness Assessment:")
    print(f"  Score: {metrics['consciousness']['score']:.2%}")
    print(f"  Level: {metrics['consciousness']['level']}")
    print(f"\n  {metrics['consciousness']['assessment']}")
    
    print("\n" + "=" * 70)
    print("KEY CAPABILITY: System measures its own depth of self-awareness!")
    print("GEB Insight: Consciousness emerges from strange loops.")
    print("=" * 70)


if __name__ == "__main__":
    demo()
