#!/usr/bin/env python3
"""
Strange Loop Optimization - Issue #24
GEB-inspired recursion control and self-reference management

Phase 4: Self-Improvement - Strange loops module
Measures recursion depth, prevents infinite loops, optimizes self-referential reasoning
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque


class LoopType(Enum):
    """Types of strange loops (GEB-inspired)"""
    SELF_REFERENCE = "self_reference"           # A refers to itself
    MUTUAL_RECURSION = "mutual_recursion"       # A ↔ B circular reference
    HIERARCHICAL_LOOP = "hierarchical_loop"     # A is-a B is-a C is-a A
    TANGLED_HIERARCHY = "tangled_hierarchy"     # Mixing levels (GEB Ch. XX)
    META_REFERENCE = "meta_reference"           # System reasoning about itself
    PRODUCTIVE_LOOP = "productive_loop"         # Loop that creates new knowledge
    INFINITE_REGRESS = "infinite_regress"       # Endless descent (problem!)


@dataclass
class RecursionEvent:
    """
    Record of entering a recursive context
    Used for detecting and managing loops
    """
    context_id: str
    recursion_level: int
    timestamp: float = field(default_factory=time.time)
    parent_context: Optional[str] = None
    concepts_involved: Set[str] = field(default_factory=set)
    operation: str = ""  # What operation triggered this recursion
    
    def __str__(self) -> str:
        indent = "  " * self.recursion_level
        return (
            f"{indent}[Level {self.recursion_level}] {self.context_id}\n"
            f"{indent}  Operation: {self.operation}\n"
            f"{indent}  Concepts: {', '.join(list(self.concepts_involved)[:3])}"
        )


@dataclass
class LoopDetectionResult:
    """
    Result of loop detection analysis
    """
    loop_detected: bool
    loop_type: Optional[LoopType] = None
    loop_path: List[str] = field(default_factory=list)
    recursion_depth: int = 0
    is_productive: bool = False  # Does loop create new knowledge?
    should_terminate: bool = False
    confidence: float = 0.0
    
    # Recommendations
    termination_reason: Optional[str] = None
    optimization_suggestion: Optional[str] = None
    
    def __str__(self) -> str:
        if not self.loop_detected:
            return "No loop detected"
        
        status = "PRODUCTIVE" if self.is_productive else "PROBLEMATIC"
        lines = [
            f"Loop Detected: {self.loop_type.value} ({status})",
            f"Depth: {self.recursion_depth}",
            f"Path: {' → '.join(self.loop_path)}",
        ]
        
        if self.should_terminate:
            lines.append(f"⚠️  TERMINATE: {self.termination_reason}")
        
        if self.optimization_suggestion:
            lines.append(f"💡 Suggestion: {self.optimization_suggestion}")
        
        return "\n".join(lines)


class StrangeLoopOptimizer:
    """
    GEB-inspired strange loop optimizer (Issue #24)
    
    Manages self-referential reasoning:
    - Detects different types of loops (productive vs infinite)
    - Measures recursion depth
    - Sets intelligent termination conditions
    - Optimizes depth vs efficiency trade-offs
    - Allows productive strange loops (GEB-style consciousness)
    """
    
    def __init__(
        self,
        max_recursion_depth: int = 10,
        max_same_context: int = 3,
        timeout_seconds: float = 5.0,
        allow_productive_loops: bool = True
    ):
        """
        Args:
            max_recursion_depth: Maximum depth before forced termination
            max_same_context: Max times to visit same context before terminating
            timeout_seconds: Max time to spend in recursive reasoning
            allow_productive_loops: Allow loops that generate new knowledge
        """
        self.max_recursion_depth = max_recursion_depth
        self.max_same_context = max_same_context
        self.timeout_seconds = timeout_seconds
        self.allow_productive_loops = allow_productive_loops
        
        # Tracking structures
        self.recursion_stack: List[RecursionEvent] = []
        self.context_visit_count: Dict[str, int] = {}
        self.loop_history: List[LoopDetectionResult] = []
        
        # Performance metrics
        self.total_loops_detected: int = 0
        self.productive_loops: int = 0
        self.terminated_loops: int = 0
        
        # Knowledge creation tracking (for productivity assessment)
        self.knowledge_before_loop: Set[str] = set()
    
    def enter_recursion(
        self,
        context_id: str,
        operation: str,
        concepts: Optional[Set[str]] = None
    ) -> RecursionEvent:
        """
        Enter a recursive context
        
        Args:
            context_id: Unique identifier for this context
            operation: What operation is being performed
            concepts: Concepts involved in this recursion
            
        Returns:
            RecursionEvent for tracking
        """
        parent_context = self.recursion_stack[-1].context_id if self.recursion_stack else None
        level = len(self.recursion_stack)
        
        event = RecursionEvent(
            context_id=context_id,
            recursion_level=level,
            parent_context=parent_context,
            concepts_involved=concepts or set(),
            operation=operation
        )
        
        self.recursion_stack.append(event)
        
        # Track visit count
        self.context_visit_count[context_id] = self.context_visit_count.get(context_id, 0) + 1
        
        return event
    
    def exit_recursion(self) -> Optional[RecursionEvent]:
        """
        Exit current recursive context
        
        Returns:
            RecursionEvent that was exited (None if stack empty)
        """
        if self.recursion_stack:
            event = self.recursion_stack.pop()
            # Don't reset visit count - keep for loop detection
            return event
        return None
    
    def check_loop(
        self,
        current_knowledge: Optional[Set[str]] = None
    ) -> LoopDetectionResult:
        """
        Check if we're in a strange loop and whether to continue
        
        Args:
            current_knowledge: Current set of known concepts (for productivity check)
            
        Returns:
            LoopDetectionResult with recommendations
        """
        if not self.recursion_stack:
            return LoopDetectionResult(loop_detected=False)
        
        current_event = self.recursion_stack[-1]
        current_depth = len(self.recursion_stack)
        current_context = current_event.context_id
        
        # Check 1: Maximum depth exceeded
        if current_depth > self.max_recursion_depth:
            result = LoopDetectionResult(
                loop_detected=True,
                loop_type=LoopType.INFINITE_REGRESS,
                recursion_depth=current_depth,
                should_terminate=True,
                termination_reason=f"Max depth {self.max_recursion_depth} exceeded",
                confidence=1.0
            )
            self.loop_history.append(result)
            self.total_loops_detected += 1
            self.terminated_loops += 1
            return result
        
        # Check 2: Same context visited too many times
        visit_count = self.context_visit_count.get(current_context, 0)
        if visit_count > self.max_same_context:
            # Check if it's productive first
            is_productive = self._is_productive_loop(current_knowledge)
            
            if not is_productive or not self.allow_productive_loops:
                result = LoopDetectionResult(
                    loop_detected=True,
                    loop_type=LoopType.SELF_REFERENCE,
                    recursion_depth=current_depth,
                    is_productive=is_productive,
                    should_terminate=True,
                    termination_reason=f"Context '{current_context}' visited {visit_count} times",
                    confidence=0.9
                )
                self.loop_history.append(result)
                self.total_loops_detected += 1
                self.terminated_loops += 1
                return result
        
        # Check 3: Circular path in stack (A → B → C → A)
        loop_path = self._find_circular_path()
        if loop_path:
            loop_type = self._classify_loop(loop_path)
            is_productive = self._is_productive_loop(current_knowledge)
            
            should_terminate = (
                not is_productive or
                not self.allow_productive_loops or
                len(loop_path) > 5  # Very long cycles are suspicious
            )
            
            result = LoopDetectionResult(
                loop_detected=True,
                loop_type=loop_type,
                loop_path=loop_path,
                recursion_depth=current_depth,
                is_productive=is_productive,
                should_terminate=should_terminate,
                termination_reason="Circular reference detected" if should_terminate else None,
                optimization_suggestion=self._suggest_optimization(loop_type, loop_path),
                confidence=0.8
            )
            
            self.loop_history.append(result)
            self.total_loops_detected += 1
            if is_productive:
                self.productive_loops += 1
            if should_terminate:
                self.terminated_loops += 1
            
            return result
        
        # Check 4: Timeout (in case of infinite computation)
        if self.recursion_stack:
            first_event = self.recursion_stack[0]
            elapsed = time.time() - first_event.timestamp
            
            if elapsed > self.timeout_seconds:
                result = LoopDetectionResult(
                    loop_detected=True,
                    loop_type=LoopType.INFINITE_REGRESS,
                    recursion_depth=current_depth,
                    should_terminate=True,
                    termination_reason=f"Timeout after {elapsed:.2f}s",
                    confidence=0.95
                )
                self.loop_history.append(result)
                self.total_loops_detected += 1
                self.terminated_loops += 1
                return result
        
        # No problematic loop detected
        return LoopDetectionResult(loop_detected=False)
    
    def _find_circular_path(self) -> List[str]:
        """
        Find if there's a circular path in the recursion stack
        Returns the circular path if found, empty list otherwise
        """
        if len(self.recursion_stack) < 2:
            return []
        
        # Check if current context appears earlier in stack
        current_context = self.recursion_stack[-1].context_id
        
        for i, event in enumerate(self.recursion_stack[:-1]):
            if event.context_id == current_context:
                # Found a cycle! Extract path from i to end
                cycle = [e.context_id for e in self.recursion_stack[i:]]
                return cycle
        
        return []
    
    def _classify_loop(self, loop_path: List[str]) -> LoopType:
        """
        Classify the type of strange loop
        """
        if len(loop_path) == 1:
            return LoopType.SELF_REFERENCE
        
        if len(loop_path) == 2:
            return LoopType.MUTUAL_RECURSION
        
        # Check if it involves meta-reasoning (context contains "meta", "self", etc.)
        meta_keywords = {'meta', 'self', 'introspect', 'reflect'}
        if any(any(kw in ctx.lower() for kw in meta_keywords) for ctx in loop_path):
            return LoopType.META_REFERENCE
        
        # Check if it's hierarchical (is-a, subtype, etc.)
        hierarchical_keywords = {'is_a', 'subtype', 'parent', 'child', 'hierarchy'}
        if any(any(kw in ctx.lower() for kw in hierarchical_keywords) for ctx in loop_path):
            return LoopType.HIERARCHICAL_LOOP
        
        # Default: tangled hierarchy (most GEB-like!)
        return LoopType.TANGLED_HIERARCHY
    
    def _is_productive_loop(self, current_knowledge: Optional[Set[str]]) -> bool:
        """
        Determine if a loop is productive (creates new knowledge)
        GEB insight: Some loops are consciousness-creating!
        """
        if current_knowledge is None:
            # Can't determine productivity without knowledge state
            return False
        
        # Check if knowledge has grown since entering the loop
        new_knowledge = current_knowledge - self.knowledge_before_loop
        
        if len(new_knowledge) > 0:
            # Loop generated new knowledge - potentially productive!
            return True
        
        # Check if we're in meta-reasoning (often productive)
        if self.recursion_stack:
            current_op = self.recursion_stack[-1].operation
            meta_ops = {'introspect', 'reflect', 'meta_reason', 'self_model', 'synthesize'}
            if any(op in current_op.lower() for op in meta_ops):
                return True
        
        return False
    
    def _suggest_optimization(self, loop_type: LoopType, loop_path: List[str]) -> str:
        """
        Suggest how to optimize this loop
        """
        if loop_type == LoopType.SELF_REFERENCE:
            return "Add base case or caching to prevent redundant computation"
        
        if loop_type == LoopType.MUTUAL_RECURSION:
            return "Consider breaking circular dependency or using memoization"
        
        if loop_type == LoopType.HIERARCHICAL_LOOP:
            return "Restructure hierarchy to remove circular is-a relations"
        
        if loop_type == LoopType.META_REFERENCE:
            return "This may be productive - allow but monitor depth"
        
        if loop_type == LoopType.TANGLED_HIERARCHY:
            return "Separate levels or use explicit level markers (GEB-style)"
        
        return "Monitor and limit recursion depth"
    
    def set_knowledge_baseline(self, knowledge: Set[str]):
        """
        Set baseline knowledge before entering a potentially loopy operation
        Used to detect productive loops
        """
        self.knowledge_before_loop = knowledge.copy()
    
    def reset(self):
        """
        Reset optimizer state (e.g., between queries)
        """
        self.recursion_stack.clear()
        self.context_visit_count.clear()
        self.knowledge_before_loop.clear()
    
    def get_current_depth(self) -> int:
        """Get current recursion depth"""
        return len(self.recursion_stack)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about strange loop optimization
        """
        total_checked = len(self.loop_history)
        
        loop_types_count = {}
        for result in self.loop_history:
            if result.loop_type:
                loop_types_count[result.loop_type.value] = \
                    loop_types_count.get(result.loop_type.value, 0) + 1
        
        return {
            'total_loops_detected': self.total_loops_detected,
            'productive_loops': self.productive_loops,
            'terminated_loops': self.terminated_loops,
            'max_depth_seen': max((r.recursion_depth for r in self.loop_history), default=0),
            'loop_types': loop_types_count,
            'productivity_rate': (
                self.productive_loops / self.total_loops_detected
                if self.total_loops_detected > 0 else 0.0
            ),
            'termination_rate': (
                self.terminated_loops / self.total_loops_detected
                if self.total_loops_detected > 0 else 0.0
            )
        }
    
    def visualize_recursion_stack(self) -> str:
        """
        Generate ASCII visualization of current recursion stack
        """
        if not self.recursion_stack:
            return "Recursion stack: EMPTY"
        
        lines = ["Recursion Stack:"]
        for i, event in enumerate(self.recursion_stack):
            indent = "  " * i
            arrow = "→ " if i > 0 else ""
            visits = self.context_visit_count.get(event.context_id, 0)
            lines.append(f"{indent}{arrow}{event.context_id} [depth={i}, visits={visits}]")
            lines.append(f"{indent}  op: {event.operation}")
        
        return "\n".join(lines)


def demo():
    """Demonstrate strange loop optimization"""
    print("=" * 70)
    print("STRANGE LOOP OPTIMIZATION - Issue #24 Demo")
    print("GEB-inspired recursion control and self-reference management")
    print("=" * 70)
    
    optimizer = StrangeLoopOptimizer(
        max_recursion_depth=5,
        max_same_context=2,
        timeout_seconds=5.0,
        allow_productive_loops=True
    )
    
    print("\n1. SIMPLE SELF-REFERENCE")
    print("-" * 70)
    
    # Simulate self-referential reasoning
    optimizer.enter_recursion("concept_A", "infer_properties", {"A"})
    print(optimizer.visualize_recursion_stack())
    
    result = optimizer.check_loop()
    print(f"\n{result}")
    
    # Visit same context again - should detect loop
    optimizer.enter_recursion("concept_A", "infer_properties", {"A"})
    result = optimizer.check_loop()
    print(f"\n{result}")
    
    optimizer.reset()
    
    print("\n\n2. MUTUAL RECURSION (A ↔ B)")
    print("-" * 70)
    
    optimizer.enter_recursion("concept_A", "find_relations", {"A", "B"})
    print(optimizer.visualize_recursion_stack())
    
    optimizer.enter_recursion("concept_B", "find_relations", {"B", "A"})
    print(optimizer.visualize_recursion_stack())
    
    # Return to A - circular!
    optimizer.enter_recursion("concept_A", "find_relations", {"A", "B"})
    result = optimizer.check_loop()
    print(f"\n{result}")
    
    optimizer.reset()
    
    print("\n\n3. HIERARCHICAL LOOP (A is-a B is-a C is-a A)")
    print("-" * 70)
    
    optimizer.enter_recursion("is_a_check_A", "check_subtype", {"A"})
    optimizer.enter_recursion("is_a_check_B", "check_subtype", {"B"})
    optimizer.enter_recursion("is_a_check_C", "check_subtype", {"C"})
    optimizer.enter_recursion("is_a_check_A", "check_subtype", {"A"})  # Back to A!
    
    print(optimizer.visualize_recursion_stack())
    result = optimizer.check_loop()
    print(f"\n{result}")
    
    optimizer.reset()
    
    print("\n\n4. PRODUCTIVE LOOP (Creates new knowledge)")
    print("-" * 70)
    
    # Set baseline
    initial_knowledge = {"concept_A", "concept_B"}
    optimizer.set_knowledge_baseline(initial_knowledge)
    
    optimizer.enter_recursion("meta_reason", "synthesize_concept", {"A", "B"})
    optimizer.enter_recursion("meta_reason", "synthesize_concept", {"A", "B"})
    
    # Knowledge grew!
    current_knowledge = {"concept_A", "concept_B", "concept_AB"}  # Synthesized new concept
    
    result = optimizer.check_loop(current_knowledge)
    print(f"{result}")
    
    if result.is_productive:
        print("\n✓ Productive loop allowed - generating new knowledge!")
    
    optimizer.reset()
    
    print("\n\n5. TANGLED HIERARCHY (GEB-style)")
    print("-" * 70)
    
    # Simulate GEB-style level confusion
    optimizer.enter_recursion("level_0_object", "reason", {"object"})
    optimizer.enter_recursion("level_1_meta", "meta_reason", {"meta"})
    optimizer.enter_recursion("level_2_meta_meta", "meta_meta_reason", {"meta_meta"})
    optimizer.enter_recursion("level_0_object", "reason", {"object"})  # Jump back down!
    
    print(optimizer.visualize_recursion_stack())
    result = optimizer.check_loop()
    print(f"\n{result}")
    
    print("\n\n6. DEPTH LIMIT PROTECTION")
    print("-" * 70)
    
    optimizer.reset()
    print("Simulating deep recursion...")
    
    for i in range(7):  # Max is 5
        optimizer.enter_recursion(f"level_{i}", f"operation_{i}", {f"concept_{i}"})
        result = optimizer.check_loop()
        
        if result.should_terminate:
            print(f"\n⚠️  TERMINATED at depth {result.recursion_depth}")
            print(f"   Reason: {result.termination_reason}")
            break
    
    print("\n\n7. OVERALL STATISTICS")
    print("-" * 70)
    
    stats = optimizer.get_statistics()
    print(f"Total loops detected: {stats['total_loops_detected']}")
    print(f"Productive loops: {stats['productive_loops']}")
    print(f"Terminated loops: {stats['terminated_loops']}")
    print(f"Max depth seen: {stats['max_depth_seen']}")
    print(f"Productivity rate: {stats['productivity_rate']:.2%}")
    print(f"Termination rate: {stats['termination_rate']:.2%}")
    
    if stats['loop_types']:
        print(f"\nLoop types encountered:")
        for loop_type, count in stats['loop_types'].items():
            print(f"  - {loop_type}: {count}")
    
    print("\n" + "=" * 70)
    print("KEY CAPABILITY: System prevents infinite loops while allowing")
    print("productive self-reference (GEB-style consciousness)!")
    print("=" * 70)


if __name__ == "__main__":
    demo()
