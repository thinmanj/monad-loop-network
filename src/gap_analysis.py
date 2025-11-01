#!/usr/bin/env python3
"""
Gap Analysis - Issue #20
Analyzes failure patterns to identify missing knowledge, relations, and inference rules

Phase 4: Self-Improvement - Diagnostic module
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from .failure_detection import FailureReport, FailureType


@dataclass
class KnowledgeGap:
    """
    Represents a gap in the knowledge base identified from failures
    """
    gap_type: str  # 'concept', 'relation', 'rule', 'pattern'
    priority: int  # 1 (critical) to 5 (low)
    description: str
    
    # What's missing
    missing_concepts: Set[str] = field(default_factory=set)
    missing_relations: List[Tuple[str, str, str]] = field(default_factory=list)  # (source, rel, target)
    missing_rules: List[str] = field(default_factory=list)
    
    # Evidence
    supporting_failures: List[FailureReport] = field(default_factory=list)
    frequency: int = 0  # How often this gap appears
    
    # Actionable recommendations
    suggested_action: str = ""
    estimated_impact: float = 0.0  # 0.0 to 1.0
    
    def __str__(self) -> str:
        """Human-readable gap report"""
        lines = [
            f"Gap Type: {self.gap_type.upper()}",
            f"Priority: P{self.priority}",
            f"Description: {self.description}",
            f"Frequency: {self.frequency} occurrences",
        ]
        
        if self.missing_concepts:
            lines.append(f"Missing Concepts: {', '.join(list(self.missing_concepts)[:5])}")
        
        if self.missing_relations:
            lines.append(f"Missing Relations: {len(self.missing_relations)} identified")
            for src, rel, tgt in self.missing_relations[:3]:
                lines.append(f"  - {src} --[{rel}]--> {tgt}")
        
        if self.missing_rules:
            lines.append(f"Missing Rules: {', '.join(self.missing_rules[:3])}")
        
        if self.suggested_action:
            lines.append(f"Action: {self.suggested_action}")
        
        if self.estimated_impact > 0:
            lines.append(f"Impact: {self.estimated_impact:.1%} of failures")
        
        return "\n".join(lines)


@dataclass
class GapAnalysisReport:
    """
    Complete analysis of knowledge gaps
    """
    gaps: List[KnowledgeGap] = field(default_factory=list)
    total_failures_analyzed: int = 0
    analysis_timestamp: float = 0.0
    
    # Summary statistics
    critical_gaps: int = 0
    total_missing_concepts: int = 0
    total_missing_relations: int = 0
    
    def get_priority_gaps(self, max_priority: int = 2) -> List[KnowledgeGap]:
        """Get high-priority gaps (P1, P2, etc.)"""
        return [g for g in self.gaps if g.priority <= max_priority]
    
    def get_by_type(self, gap_type: str) -> List[KnowledgeGap]:
        """Get gaps of specific type"""
        return [g for g in self.gaps if g.gap_type == gap_type]
    
    def __str__(self) -> str:
        """Human-readable analysis report"""
        lines = [
            "=" * 70,
            "KNOWLEDGE GAP ANALYSIS REPORT",
            "=" * 70,
            f"Total Failures Analyzed: {self.total_failures_analyzed}",
            f"Gaps Identified: {len(self.gaps)}",
            f"Critical Gaps (P1-P2): {self.critical_gaps}",
            f"Missing Concepts: {self.total_missing_concepts}",
            f"Missing Relations: {self.total_missing_relations}",
            "",
        ]
        
        if self.gaps:
            lines.append("Top Priority Gaps:")
            lines.append("-" * 70)
            for i, gap in enumerate(self.get_priority_gaps(max_priority=2)[:5], 1):
                lines.append(f"\n{i}. {gap}\n")
        
        return "\n".join(lines)


class GapAnalyzer:
    """
    Analyzes failure patterns to identify knowledge gaps (Issue #20)
    """
    
    def __init__(self, min_frequency: int = 2):
        """
        Args:
            min_frequency: Minimum occurrences to consider a gap significant
        """
        self.min_frequency = min_frequency
    
    def analyze_failures(
        self,
        failures: List[FailureReport],
        knowledge_graph: Optional[Dict] = None
    ) -> GapAnalysisReport:
        """
        Analyze failure reports to identify knowledge gaps
        
        Args:
            failures: List of failure reports
            knowledge_graph: Current knowledge graph for context
            
        Returns:
            Complete gap analysis report
        """
        if not failures:
            return GapAnalysisReport(total_failures_analyzed=0)
        
        gaps = []
        
        # Analyze missing concepts
        concept_gaps = self._analyze_missing_concepts(failures)
        gaps.extend(concept_gaps)
        
        # Analyze missing relations
        relation_gaps = self._analyze_missing_relations(failures, knowledge_graph)
        gaps.extend(relation_gaps)
        
        # Analyze inference failures
        rule_gaps = self._analyze_inference_failures(failures)
        gaps.extend(rule_gaps)
        
        # Analyze structural patterns
        pattern_gaps = self._analyze_structural_patterns(failures)
        gaps.extend(pattern_gaps)
        
        # Sort by priority and impact
        gaps.sort(key=lambda g: (g.priority, -g.frequency))
        
        # Build report
        report = GapAnalysisReport(
            gaps=gaps,
            total_failures_analyzed=len(failures),
            critical_gaps=len([g for g in gaps if g.priority <= 2]),
            total_missing_concepts=sum(len(g.missing_concepts) for g in gaps),
            total_missing_relations=sum(len(g.missing_relations) for g in gaps)
        )
        
        return report
    
    def _analyze_missing_concepts(
        self,
        failures: List[FailureReport]
    ) -> List[KnowledgeGap]:
        """Identify missing concepts from MISSING_CONCEPT failures"""
        gaps = []
        
        # Collect all missing concepts
        concept_frequency = Counter()
        concept_failures = defaultdict(list)
        
        for failure in failures:
            if failure.failure_type == FailureType.MISSING_CONCEPT:
                for concept in failure.missing_concepts:
                    # Filter out stopwords and punctuation
                    if len(concept) > 2 and concept.isalnum():
                        concept_frequency[concept] += 1
                        concept_failures[concept].append(failure)
        
        # Create gaps for frequent missing concepts
        for concept, freq in concept_frequency.items():
            if freq >= self.min_frequency:
                gap = KnowledgeGap(
                    gap_type='concept',
                    priority=self._calculate_priority(freq, len(failures)),
                    description=f"Concept '{concept}' frequently missing from queries",
                    missing_concepts={concept},
                    supporting_failures=concept_failures[concept],
                    frequency=freq,
                    suggested_action=f"Add concept '{concept}' to knowledge base",
                    estimated_impact=freq / len(failures)
                )
                gaps.append(gap)
        
        return gaps
    
    def _analyze_missing_relations(
        self,
        failures: List[FailureReport],
        knowledge_graph: Optional[Dict]
    ) -> List[KnowledgeGap]:
        """Identify missing relations from INCOMPLETE_PATH failures"""
        gaps = []
        
        # Collect patterns of incomplete paths
        incomplete_paths = [f for f in failures if f.failure_type == FailureType.INCOMPLETE_PATH]
        
        if incomplete_paths:
            gap = KnowledgeGap(
                gap_type='relation',
                priority=2,
                description=f"{len(incomplete_paths)} queries failed due to missing connections",
                supporting_failures=incomplete_paths,
                frequency=len(incomplete_paths),
                suggested_action="Add relations to connect isolated concepts",
                estimated_impact=len(incomplete_paths) / len(failures)
            )
            
            # Try to infer missing relations from query context
            for failure in incomplete_paths[:5]:  # Sample
                # Simple heuristic: extract potential relation from query
                words = failure.query.lower().split()
                if 'is' in words:
                    # Potential is_a relation
                    gap.missing_relations.append(("unknown", "is_a", "unknown"))
                elif 'has' in words:
                    gap.missing_relations.append(("unknown", "has", "unknown"))
            
            gaps.append(gap)
        
        return gaps
    
    def _analyze_inference_failures(
        self,
        failures: List[FailureReport]
    ) -> List[KnowledgeGap]:
        """Identify missing inference rules from WRONG_INFERENCE failures"""
        gaps = []
        
        wrong_inferences = [f for f in failures if f.failure_type == FailureType.WRONG_INFERENCE]
        
        if wrong_inferences:
            # Analyze patterns in wrong inferences
            gap = KnowledgeGap(
                gap_type='rule',
                priority=1,  # High priority - affects correctness
                description=f"{len(wrong_inferences)} queries produced incorrect inferences",
                supporting_failures=wrong_inferences,
                frequency=len(wrong_inferences),
                suggested_action="Review and update inference rules",
                estimated_impact=len(wrong_inferences) / len(failures)
            )
            
            # Suggest specific rule improvements
            gap.missing_rules.append("Add type-checking constraints")
            gap.missing_rules.append("Add domain-specific inference rules")
            
            gaps.append(gap)
        
        # Check for circular reasoning
        circular = [f for f in failures if f.failure_type == FailureType.CIRCULAR_REASONING]
        if circular:
            gap = KnowledgeGap(
                gap_type='rule',
                priority=1,
                description=f"{len(circular)} queries resulted in circular reasoning",
                supporting_failures=circular,
                frequency=len(circular),
                suggested_action="Add loop detection and termination conditions",
                estimated_impact=len(circular) / len(failures)
            )
            gap.missing_rules.append("Loop detection mechanism")
            gap.missing_rules.append("Maximum inference depth limit")
            gaps.append(gap)
        
        return gaps
    
    def _analyze_structural_patterns(
        self,
        failures: List[FailureReport]
    ) -> List[KnowledgeGap]:
        """Identify missing structural patterns"""
        gaps = []
        
        # Look for patterns in low-confidence results
        low_confidence = [f for f in failures if f.failure_type == FailureType.LOW_CONFIDENCE]
        
        if low_confidence:
            gap = KnowledgeGap(
                gap_type='pattern',
                priority=3,
                description=f"{len(low_confidence)} queries had low confidence scores",
                supporting_failures=low_confidence,
                frequency=len(low_confidence),
                suggested_action="Enrich knowledge base with more structural patterns",
                estimated_impact=len(low_confidence) / len(failures)
            )
            gaps.append(gap)
        
        # Look for ambiguous queries
        ambiguous = [f for f in failures if f.failure_type == FailureType.AMBIGUOUS_QUERY]
        if ambiguous:
            gap = KnowledgeGap(
                gap_type='pattern',
                priority=3,
                description=f"{len(ambiguous)} queries were ambiguous",
                supporting_failures=ambiguous,
                frequency=len(ambiguous),
                suggested_action="Add disambiguation patterns or request clarification",
                estimated_impact=len(ambiguous) / len(failures)
            )
            gaps.append(gap)
        
        return gaps
    
    def _calculate_priority(self, frequency: int, total_failures: int) -> int:
        """
        Calculate priority based on frequency
        1 = Critical (>50% of failures)
        2 = High (>25%)
        3 = Medium (>10%)
        4 = Low (>5%)
        5 = Very Low (<5%)
        """
        ratio = frequency / max(total_failures, 1)
        
        if ratio > 0.5:
            return 1
        elif ratio > 0.25:
            return 2
        elif ratio > 0.10:
            return 3
        elif ratio > 0.05:
            return 4
        else:
            return 5
    
    def suggest_improvements(
        self,
        gap: KnowledgeGap,
        knowledge_graph: Optional[Dict] = None
    ) -> List[str]:
        """
        Generate specific improvement suggestions for a gap
        
        Returns:
            List of actionable improvement steps
        """
        suggestions = []
        
        if gap.gap_type == 'concept':
            suggestions.append(f"Add missing concepts: {', '.join(list(gap.missing_concepts)[:5])}")
            suggestions.append("Use ontology loader to import related concepts")
            suggestions.append("Learn concept definitions from examples")
        
        elif gap.gap_type == 'relation':
            suggestions.append("Add missing relations between concepts")
            suggestions.append("Use analogical reasoning to infer likely relations")
            suggestions.append("Import relations from external ontologies")
        
        elif gap.gap_type == 'rule':
            suggestions.append("Review and update inference rules")
            suggestions.append("Add domain-specific constraints")
            suggestions.append("Implement missing inference patterns")
        
        elif gap.gap_type == 'pattern':
            suggestions.append("Enrich knowledge base with structural patterns")
            suggestions.append("Use pattern learning to extract common structures")
            suggestions.append("Add disambiguation mechanisms")
        
        return suggestions


def demo_gap_analysis():
    """Demonstrate gap analysis capabilities"""
    print("=" * 70)
    print("GAP ANALYSIS DEMO - Issue #20")
    print("Identifying knowledge gaps from failures")
    print("=" * 70)
    print()
    
    # Import after function definition to avoid circular import
    from .failure_detection import FailureDetector, FailureReport, FailureType
    
    # Create mock failures
    failures = [
        FailureReport(
            failure_type=FailureType.MISSING_CONCEPT,
            query="What is a cat?",
            missing_concepts={'cat'},
            suggested_fix="Add concept: cat"
        ),
        FailureReport(
            failure_type=FailureType.MISSING_CONCEPT,
            query="Tell me about cats",
            missing_concepts={'cat'},
            suggested_fix="Add concept: cat"
        ),
        FailureReport(
            failure_type=FailureType.MISSING_CONCEPT,
            query="Are cats animals?",
            missing_concepts={'cat'},
            suggested_fix="Add concept: cat"
        ),
        FailureReport(
            failure_type=FailureType.INCOMPLETE_PATH,
            query="How is a dog related to a wolf?",
            suggested_fix="Add relations"
        ),
        FailureReport(
            failure_type=FailureType.INCOMPLETE_PATH,
            query="Connect mammal to vertebrate",
            suggested_fix="Add relations"
        ),
        FailureReport(
            failure_type=FailureType.WRONG_INFERENCE,
            query="Is a whale a fish?",
            actual_result={'answer': 'yes'},
            suggested_fix="Fix inference rules"
        ),
        FailureReport(
            failure_type=FailureType.CIRCULAR_REASONING,
            query="Prove A",
            inference_chain=['A→B', 'B→C', 'C→A'],
            suggested_fix="Add loop detection"
        ),
        FailureReport(
            failure_type=FailureType.LOW_CONFIDENCE,
            query="Uncertain query",
            confidence_score=0.3,
            suggested_fix="Add more knowledge"
        ),
    ]
    
    print(f"Analyzing {len(failures)} failure reports...\n")
    
    # Analyze
    analyzer = GapAnalyzer(min_frequency=2)
    report = analyzer.analyze_failures(failures, knowledge_graph={'dog': {}, 'animal': {}})
    
    print(report)
    
    # Show detailed suggestions for top gap
    if report.gaps:
        print("\nDetailed Suggestions for Top Gap:")
        print("-" * 70)
        top_gap = report.gaps[0]
        suggestions = analyzer.suggest_improvements(top_gap)
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
    
    print("\n" + "=" * 70)
    print("✓ Gap analysis complete!")
    print("=" * 70)
    print()
    print("Key Capabilities:")
    print("  ✓ Identifies missing concepts with frequency analysis")
    print("  ✓ Detects incomplete relation networks")
    print("  ✓ Finds inference rule gaps")
    print("  ✓ Analyzes structural patterns")
    print("  ✓ Prioritizes gaps by impact")
    print("  ✓ Generates actionable improvement suggestions")
    print("  ✓ Ready for concept synthesis (Issue #21)")
    print("=" * 70)


if __name__ == '__main__':
    demo_gap_analysis()
