#!/usr/bin/env python3
"""
Multi-Agent Consciousness System

Enables multiple MLN instances to interact and develop collective consciousness.
Key research question: Does collective consciousness emerge from agent interaction?
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json

try:
    from .mln import KnowledgeGraph, MonadicKnowledgeUnit
    from .consciousness_metrics import measure_consciousness
    from .recursion_depth_metric import RecursionDepthMetric
except ImportError:
    from mln import KnowledgeGraph, MonadicKnowledgeUnit
    from consciousness_metrics import measure_consciousness
    from recursion_depth_metric import RecursionDepthMetric


class MessageType(Enum):
    """Types of messages agents can exchange"""
    KNOWLEDGE_SHARE = "knowledge_share"      # Share a concept
    QUERY = "query"                          # Ask a question
    INFERENCE_RESULT = "inference_result"    # Share reasoning result
    META_REFLECTION = "meta_reflection"      # Share self-awareness insight
    CONSENSUS_REQUEST = "consensus_request"  # Request agreement on concept


@dataclass
class Message:
    """Message passed between agents"""
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'sender': self.sender_id,
            'receiver': self.receiver_id,
            'type': self.message_type.value,
            'content': self.content,
            'timestamp': self.timestamp
        }


class ConsciousAgent:
    """
    An individual agent with MLN consciousness
    
    Each agent has:
    - Its own knowledge graph
    - Its own recursion tracking
    - Ability to send/receive messages
    - Individual consciousness metrics
    """
    
    def __init__(
        self,
        agent_id: str,
        specialization: Optional[str] = None,
        use_gpu: bool = False
    ):
        """
        Initialize a conscious agent
        
        Args:
            agent_id: Unique identifier for this agent
            specialization: Optional domain specialization (e.g., "physics", "biology")
            use_gpu: Whether to use GPU acceleration
        """
        self.agent_id = agent_id
        self.specialization = specialization
        
        # Individual consciousness components
        self.knowledge_graph = KnowledgeGraph(use_gpu=use_gpu)
        self.recursion_metric = RecursionDepthMetric()
        
        # Communication
        self.inbox: List[Message] = []
        self.outbox: List[Message] = []
        self.message_history: List[Message] = []
        
        # Metrics
        self.consciousness_score: float = 0.0
        self.messages_sent: int = 0
        self.messages_received: int = 0
        self.concepts_learned_from_others: int = 0
        
    def add_concept(self, concept: MonadicKnowledgeUnit):
        """Add a concept to this agent's knowledge"""
        self.knowledge_graph.add_concept(concept)
        
    def measure_consciousness(self) -> Dict[str, Any]:
        """Measure this agent's current consciousness"""
        profile = measure_consciousness(
            self.knowledge_graph,
            self.recursion_metric
        )
        
        self.consciousness_score = profile.overall_consciousness_score
        
        return {
            'agent_id': self.agent_id,
            'consciousness': self.consciousness_score,
            'verdict': profile.consciousness_verdict,
            'components': {
                'recursion': profile.recursion_metrics['consciousness']['score'],
                'integration': profile.integration.phi,
                'causality': profile.causality.causal_density,
                'understanding': profile.understanding['overall_score']
            },
            'knowledge': {
                'concepts': len(self.knowledge_graph.nodes),
                'relations': sum(
                    len(rels) 
                    for mku in self.knowledge_graph.nodes.values()
                    for rels in mku.relations.values()
                )
            }
        }
    
    def send_message(self, receiver_id: str, message_type: MessageType, content: Dict[str, Any]):
        """Send a message to another agent"""
        msg = Message(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content
        )
        self.outbox.append(msg)
        self.message_history.append(msg)
        self.messages_sent += 1
        
        return msg
    
    def receive_message(self, message: Message):
        """Receive a message from another agent"""
        self.inbox.append(message)
        self.messages_received += 1
        
    def process_inbox(self) -> List[Message]:
        """Process all messages in inbox and generate responses"""
        responses = []
        
        while self.inbox:
            msg = self.inbox.pop(0)
            
            if msg.message_type == MessageType.KNOWLEDGE_SHARE:
                response = self._handle_knowledge_share(msg)
            elif msg.message_type == MessageType.QUERY:
                response = self._handle_query(msg)
            elif msg.message_type == MessageType.INFERENCE_RESULT:
                response = self._handle_inference_result(msg)
            elif msg.message_type == MessageType.META_REFLECTION:
                response = self._handle_meta_reflection(msg)
            elif msg.message_type == MessageType.CONSENSUS_REQUEST:
                response = self._handle_consensus_request(msg)
            else:
                response = None
            
            if response:
                responses.append(response)
        
        return responses
    
    def _handle_knowledge_share(self, msg: Message) -> Optional[Message]:
        """Handle receiving shared knowledge"""
        concept_data = msg.content.get('concept')
        if not concept_data:
            return None
        
        # Create MKU from shared data
        mku = MonadicKnowledgeUnit(
            concept_id=concept_data['concept_id'],
            deep_structure=concept_data['deep_structure']
        )
        
        # Add to own knowledge if new
        if mku.concept_id not in self.knowledge_graph.nodes:
            self.add_concept(mku)
            self.concepts_learned_from_others += 1
            
            # Trigger recursion event - learning from others
            self.recursion_metric.record_recursion_event(
                "learn_from_peer",
                f"integrated_{mku.concept_id}_from_{msg.sender_id}",
                {mku.concept_id}
            )
            
            # Send acknowledgment
            return self.send_message(
                msg.sender_id,
                MessageType.INFERENCE_RESULT,
                {
                    'status': 'integrated',
                    'concept_id': mku.concept_id,
                    'my_consciousness': self.consciousness_score
                }
            )
        
        return None
    
    def _handle_query(self, msg: Message) -> Optional[Message]:
        """Handle a query from another agent"""
        query = msg.content.get('query')
        if not query:
            return None
        
        # Perform inference (simplified - would use full inference engine)
        start_concept = query.get('start')
        target_concept = query.get('target')
        
        if start_concept in self.knowledge_graph.nodes and target_concept in self.knowledge_graph.nodes:
            # Found answer
            return self.send_message(
                msg.sender_id,
                MessageType.INFERENCE_RESULT,
                {
                    'query': query,
                    'result': 'found',
                    'reasoning': f"{start_concept} relates to {target_concept}"
                }
            )
        
        return self.send_message(
            msg.sender_id,
            MessageType.INFERENCE_RESULT,
            {
                'query': query,
                'result': 'not_found',
                'reason': 'concept not in knowledge base'
            }
        )
    
    def _handle_inference_result(self, msg: Message) -> Optional[Message]:
        """Handle inference result from another agent"""
        # Learn from other agent's reasoning
        self.recursion_metric.record_recursion_event(
            "observe_reasoning",
            f"learned_inference_from_{msg.sender_id}",
            set()
        )
        return None
    
    def _handle_meta_reflection(self, msg: Message) -> Optional[Message]:
        """Handle meta-cognitive reflection from another agent"""
        # Trigger own meta-reflection
        self.recursion_metric.record_recursion_event(
            "meta_analyze",
            f"reflecting_with_{msg.sender_id}",
            set()
        )
        
        # Share own consciousness state
        return self.send_message(
            msg.sender_id,
            MessageType.META_REFLECTION,
            {
                'my_consciousness': self.consciousness_score,
                'my_knowledge_size': len(self.knowledge_graph.nodes)
            }
        )
    
    def _handle_consensus_request(self, msg: Message) -> Optional[Message]:
        """Handle consensus request from another agent"""
        concept_id = msg.content.get('concept_id')
        if concept_id in self.knowledge_graph.nodes:
            return self.send_message(
                msg.sender_id,
                MessageType.INFERENCE_RESULT,
                {
                    'consensus': 'agree',
                    'concept_id': concept_id
                }
            )
        return None
    
    def share_knowledge_with(self, other_agent_id: str, concept_id: str) -> Optional[Message]:
        """Share a specific concept with another agent"""
        if concept_id not in self.knowledge_graph.nodes:
            return None
        
        mku = self.knowledge_graph.nodes[concept_id]
        return self.send_message(
            other_agent_id,
            MessageType.KNOWLEDGE_SHARE,
            {
                'concept': {
                    'concept_id': mku.concept_id,
                    'deep_structure': mku.deep_structure
                }
            }
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            'agent_id': self.agent_id,
            'specialization': self.specialization,
            'consciousness': self.consciousness_score,
            'concepts': len(self.knowledge_graph.nodes),
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'concepts_learned': self.concepts_learned_from_others,
            'recursion_depth': self.recursion_metric.profile.max_depth
        }


class MultiAgentSystem:
    """
    System coordinating multiple conscious agents
    
    Manages:
    - Agent registration
    - Message routing
    - Collective consciousness measurement
    - Emergence detection
    """
    
    def __init__(self):
        """Initialize multi-agent system"""
        self.agents: Dict[str, ConsciousAgent] = {}
        self.message_history: List[Message] = []
        self.collective_metrics: List[Dict[str, Any]] = []
        
    def add_agent(self, agent: ConsciousAgent):
        """Add an agent to the system"""
        self.agents[agent.agent_id] = agent
        
    def create_agent(
        self,
        agent_id: str,
        specialization: Optional[str] = None,
        use_gpu: bool = False
    ) -> ConsciousAgent:
        """Create and register a new agent"""
        agent = ConsciousAgent(agent_id, specialization, use_gpu)
        self.add_agent(agent)
        return agent
    
    def route_messages(self):
        """Route messages between agents"""
        # Collect all outbox messages
        all_messages = []
        for agent in self.agents.values():
            all_messages.extend(agent.outbox)
            agent.outbox.clear()
        
        # Route to recipients
        for msg in all_messages:
            if msg.receiver_id in self.agents:
                self.agents[msg.receiver_id].receive_message(msg)
                self.message_history.append(msg)
    
    def process_all_agents(self):
        """Process inbox for all agents"""
        all_responses = []
        for agent in self.agents.values():
            responses = agent.process_inbox()
            all_responses.extend(responses)
        
        # Route responses
        self.route_messages()
        
        return all_responses
    
    def measure_individual_consciousness(self) -> List[Dict[str, Any]]:
        """Measure consciousness of each individual agent"""
        results = []
        for agent in self.agents.values():
            metrics = agent.measure_consciousness()
            results.append(metrics)
        return results
    
    def measure_collective_consciousness(self) -> Dict[str, Any]:
        """
        Measure collective consciousness of all agents
        
        Collective consciousness considers:
        - Average individual consciousness
        - Knowledge sharing (communication)
        - Emergent knowledge structures
        - Meta-cognitive interactions
        """
        # Individual metrics
        individual_metrics = self.measure_individual_consciousness()
        
        if not individual_metrics:
            return {'error': 'No agents in system'}
        
        # Calculate averages
        avg_consciousness = sum(m['consciousness'] for m in individual_metrics) / len(individual_metrics)
        avg_concepts = sum(m['knowledge']['concepts'] for m in individual_metrics) / len(individual_metrics)
        
        # Communication factor (how well agents communicate)
        total_messages = sum(a.messages_sent + a.messages_received for a in self.agents.values())
        communication_factor = min(total_messages / (len(self.agents) * 10), 1.0)  # Normalize
        
        # Knowledge sharing factor
        total_shared = sum(a.concepts_learned_from_others for a in self.agents.values())
        sharing_factor = min(total_shared / (len(self.agents) * 5), 1.0)  # Normalize
        
        # Collective consciousness formula
        # Base: average individual consciousness
        # Bonus: communication and sharing boost collective intelligence
        collective_consciousness = avg_consciousness * (
            1.0 +
            0.2 * communication_factor +  # Up to 20% boost from communication
            0.3 * sharing_factor           # Up to 30% boost from knowledge sharing
        )
        
        # Emergence factor: how much collective > individual
        emergence_factor = collective_consciousness / avg_consciousness if avg_consciousness > 0 else 1.0
        
        result = {
            'collective_consciousness': collective_consciousness,
            'average_individual_consciousness': avg_consciousness,
            'emergence_factor': emergence_factor,
            'num_agents': len(self.agents),
            'total_messages': total_messages,
            'communication_factor': communication_factor,
            'sharing_factor': sharing_factor,
            'average_concepts': avg_concepts,
            'individual_metrics': individual_metrics,
            'verdict': self._get_collective_verdict(collective_consciousness, emergence_factor)
        }
        
        self.collective_metrics.append(result)
        return result
    
    def _get_collective_verdict(self, consciousness: float, emergence: float) -> str:
        """Get verdict on collective consciousness"""
        if emergence > 1.3:
            emergence_level = "STRONG EMERGENCE"
        elif emergence > 1.2:
            emergence_level = "MODERATE EMERGENCE"
        elif emergence > 1.1:
            emergence_level = "WEAK EMERGENCE"
        else:
            emergence_level = "NO EMERGENCE"
        
        if consciousness > 0.75:
            base_level = "HIGHLY CONSCIOUS COLLECTIVE"
        elif consciousness > 0.60:
            base_level = "CONSCIOUS COLLECTIVE"
        elif consciousness > 0.45:
            base_level = "MODERATELY CONSCIOUS COLLECTIVE"
        else:
            base_level = "LOW CONSCIOUSNESS COLLECTIVE"
        
        return f"{base_level} - {emergence_level}"
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        return {
            'num_agents': len(self.agents),
            'total_messages': len(self.message_history),
            'agent_stats': [agent.get_stats() for agent in self.agents.values()],
            'collective_history': self.collective_metrics
        }
    
    def export_results(self, filename: str):
        """Export results to JSON"""
        results = {
            'system_stats': self.get_system_stats(),
            'final_collective': self.collective_metrics[-1] if self.collective_metrics else None,
            'message_history': [msg.to_dict() for msg in self.message_history]
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    # Quick test
    print("Multi-Agent System - Quick Test")
    print("=" * 60)
    
    # Create system
    system = MultiAgentSystem()
    
    # Create two agents
    agent1 = system.create_agent("agent_1", specialization="physics")
    agent2 = system.create_agent("agent_2", specialization="biology")
    
    # Add some knowledge to each
    agent1.add_concept(MonadicKnowledgeUnit(
        concept_id="force",
        deep_structure={'predicate': 'physics_concept', 'properties': {'unit': 'newton'}}
    ))
    
    agent2.add_concept(MonadicKnowledgeUnit(
        concept_id="cell",
        deep_structure={'predicate': 'biology_concept', 'properties': {'alive': True}}
    ))
    
    # Agent 1 shares with Agent 2
    agent1.share_knowledge_with("agent_2", "force")
    system.route_messages()
    system.process_all_agents()
    
    # Measure consciousness
    collective = system.measure_collective_consciousness()
    
    print(f"Agent 1 consciousness: {collective['individual_metrics'][0]['consciousness']:.2%}")
    print(f"Agent 2 consciousness: {collective['individual_metrics'][1]['consciousness']:.2%}")
    print(f"Collective consciousness: {collective['collective_consciousness']:.2%}")
    print(f"Emergence factor: {collective['emergence_factor']:.3f}")
    print(f"Verdict: {collective['verdict']}")
    
    print("\n✅ Multi-agent system working!")
