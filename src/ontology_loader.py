#!/usr/bin/env python3
"""
Ontology Loader - Issue #13
Import knowledge from external ontologies (ConceptNet, DBpedia, Wikidata)
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import requests
import json
import time


@dataclass
class OntologyTriple:
    """A single fact from an ontology"""
    subject: str
    relation: str
    object: str
    source: str  # 'conceptnet', 'dbpedia', 'wikidata'
    confidence: float = 1.0


class ConceptNetLoader:
    """
    Load knowledge from ConceptNet
    https://conceptnet.io/
    
    ConceptNet is a semantic network with common-sense knowledge
    """
    
    def __init__(self, base_url: str = "http://api.conceptnet.io"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_concept(self, concept: str, limit: int = 20) -> List[OntologyTriple]:
        """
        Get edges (relations) for a concept
        
        Args:
            concept: Concept name (e.g., 'dog', 'animal')
            limit: Maximum number of edges to retrieve
            
        Returns:
            List of OntologyTriple facts
        """
        # Format concept for ConceptNet API
        if not concept.startswith('/c/en/'):
            concept_uri = f"/c/en/{concept.lower().replace(' ', '_')}"
        else:
            concept_uri = concept
        
        url = f"{self.base_url}{concept_uri}"
        params = {'limit': limit}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            triples = []
            for edge in data.get('edges', []):
                # Extract relation
                rel = edge.get('rel', {}).get('@id', '').split('/')[-1]
                
                # Extract subject and object
                start = edge.get('start', {}).get('@id', '').split('/c/en/')[-1]
                end = edge.get('end', {}).get('@id', '').split('/c/en/')[-1]
                
                # Get confidence weight
                weight = edge.get('weight', 1.0)
                
                if start and end and rel:
                    triples.append(OntologyTriple(
                        subject=start.replace('_', ' '),
                        relation=rel,
                        object=end.replace('_', ' '),
                        source='conceptnet',
                        confidence=weight
                    ))
            
            return triples
            
        except requests.RequestException as e:
            print(f"Warning: Could not fetch from ConceptNet: {e}")
            return []
    
    def search_concepts(self, query: str, limit: int = 10) -> List[str]:
        """Search for concepts matching a query"""
        url = f"{self.base_url}/search"
        params = {'text': query, 'limit': limit}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            concepts = []
            for edge in data.get('edges', []):
                start = edge.get('start', {}).get('@id', '').split('/c/en/')[-1]
                end = edge.get('end', {}).get('@id', '').split('/c/en/')[-1]
                
                if start:
                    concepts.append(start.replace('_', ' '))
                if end:
                    concepts.append(end.replace('_', ' '))
            
            return list(set(concepts))[:limit]
            
        except requests.RequestException as e:
            print(f"Warning: Could not search ConceptNet: {e}")
            return []


class DBpediaLoader:
    """
    Load knowledge from DBpedia
    https://www.dbpedia.org/
    
    DBpedia extracts structured information from Wikipedia
    """
    
    def __init__(self, sparql_endpoint: str = "http://dbpedia.org/sparql"):
        self.endpoint = sparql_endpoint
        self.session = requests.Session()
    
    def get_concept_info(self, concept: str, limit: int = 20) -> List[OntologyTriple]:
        """
        Get facts about a concept from DBpedia using SPARQL
        
        Args:
            concept: Concept name (e.g., 'Dog', 'Animal')
            limit: Maximum number of triples
            
        Returns:
            List of OntologyTriple facts
        """
        # Format resource URI
        resource_uri = f"http://dbpedia.org/resource/{concept.replace(' ', '_')}"
        
        # SPARQL query to get properties
        query = f"""
        SELECT DISTINCT ?p ?o WHERE {{
            <{resource_uri}> ?p ?o .
            FILTER(isLiteral(?o) || regex(str(?o), "dbpedia.org/resource"))
        }}
        LIMIT {limit}
        """
        
        try:
            response = self.session.get(
                self.endpoint,
                params={'query': query, 'format': 'json'},
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            triples = []
            for binding in data.get('results', {}).get('bindings', []):
                predicate = binding.get('p', {}).get('value', '')
                obj = binding.get('o', {}).get('value', '')
                
                # Extract predicate name
                if '/' in predicate:
                    rel = predicate.split('/')[-1]
                else:
                    rel = predicate
                
                # Extract object value
                if 'dbpedia.org/resource/' in obj:
                    obj = obj.split('resource/')[-1].replace('_', ' ')
                
                if rel and obj and len(obj) < 100:  # Skip very long literals
                    triples.append(OntologyTriple(
                        subject=concept,
                        relation=rel,
                        object=obj,
                        source='dbpedia',
                        confidence=1.0
                    ))
            
            return triples
            
        except requests.RequestException as e:
            print(f"Warning: Could not fetch from DBpedia: {e}")
            return []


class WikidataLoader:
    """
    Load knowledge from Wikidata
    https://www.wikidata.org/
    
    Wikidata is a structured knowledge base
    """
    
    def __init__(self, api_url: str = "https://www.wikidata.org/w/api.php"):
        self.api_url = api_url
        self.session = requests.Session()
    
    def search_entity(self, concept: str) -> Optional[str]:
        """Search for an entity ID"""
        params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'language': 'en',
            'search': concept,
            'limit': 1
        }
        
        try:
            response = self.session.get(self.api_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('search'):
                return data['search'][0]['id']
            return None
            
        except requests.RequestException as e:
            print(f"Warning: Could not search Wikidata: {e}")
            return None
    
    def get_concept_info(self, concept: str, limit: int = 20) -> List[OntologyTriple]:
        """Get facts about a concept from Wikidata"""
        entity_id = self.search_entity(concept)
        if not entity_id:
            return []
        
        params = {
            'action': 'wbgetentities',
            'format': 'json',
            'ids': entity_id,
            'props': 'claims'
        }
        
        try:
            response = self.session.get(self.api_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            entity = data.get('entities', {}).get(entity_id, {})
            claims = entity.get('claims', {})
            
            triples = []
            for prop_id, claim_list in list(claims.items())[:limit]:
                for claim in claim_list[:1]:  # Take first claim for each property
                    mainsnak = claim.get('mainsnak', {})
                    datavalue = mainsnak.get('datavalue', {})
                    
                    if datavalue.get('type') == 'wikibase-entityid':
                        obj_id = datavalue.get('value', {}).get('id', '')
                        # Would need another API call to get label, skip for now
                        continue
                    elif datavalue.get('type') == 'string':
                        obj = datavalue.get('value', '')
                    else:
                        continue
                    
                    if obj and len(obj) < 100:
                        triples.append(OntologyTriple(
                            subject=concept,
                            relation=prop_id,
                            object=obj,
                            source='wikidata',
                            confidence=1.0
                        ))
            
            return triples
            
        except requests.RequestException as e:
            print(f"Warning: Could not fetch from Wikidata: {e}")
            return []


class OntologyIntegrator:
    """
    Integrate multiple ontology sources
    Converts ontology triples to MonadicKnowledgeUnits
    """
    
    def __init__(self):
        self.conceptnet = ConceptNetLoader()
        self.dbpedia = DBpediaLoader()
        self.wikidata = WikidataLoader()
    
    def load_concept(
        self,
        concept: str,
        sources: List[str] = ['conceptnet'],
        limit_per_source: int = 20
    ) -> List[OntologyTriple]:
        """
        Load a concept from multiple sources
        
        Args:
            concept: Concept to load
            sources: List of sources to use ('conceptnet', 'dbpedia', 'wikidata')
            limit_per_source: Max triples per source
            
        Returns:
            Combined list of OntologyTriple facts
        """
        all_triples = []
        
        if 'conceptnet' in sources:
            print(f"  Loading from ConceptNet...")
            triples = self.conceptnet.get_concept(concept, limit_per_source)
            all_triples.extend(triples)
            print(f"    Found {len(triples)} facts")
        
        if 'dbpedia' in sources:
            print(f"  Loading from DBpedia...")
            triples = self.dbpedia.get_concept_info(concept, limit_per_source)
            all_triples.extend(triples)
            print(f"    Found {len(triples)} facts")
        
        if 'wikidata' in sources:
            print(f"  Loading from Wikidata...")
            triples = self.wikidata.get_concept_info(concept, limit_per_source)
            all_triples.extend(triples)
            print(f"    Found {len(triples)} facts")
        
        return all_triples
    
    def triples_to_mku_data(
        self,
        triples: List[OntologyTriple],
        min_confidence: float = 0.5
    ) -> Dict[str, Dict]:
        """
        Convert ontology triples to MKU data structures
        
        Args:
            triples: List of OntologyTriple facts
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dict mapping concept_id -> deep_structure
        """
        # Filter by confidence
        triples = [t for t in triples if t.confidence >= min_confidence]
        
        # Group triples by subject
        concept_data = {}
        
        for triple in triples:
            concept_id = triple.subject
            
            if concept_id not in concept_data:
                concept_data[concept_id] = {
                    'predicate': 'ontology_concept',
                    'properties': {},
                    'relations': {},
                    'source': triple.source
                }
            
            # Classify relation type
            rel = triple.relation.lower()
            
            # Property relations (store as properties)
            if rel in ['definition', 'description', 'label', 'name']:
                concept_data[concept_id]['properties'][rel] = triple.object
            
            # Relational edges (store as relations)
            else:
                if rel not in concept_data[concept_id]['relations']:
                    concept_data[concept_id]['relations'][rel] = set()
                concept_data[concept_id]['relations'][rel].add(triple.object)
        
        # Convert sets to lists for JSON serialization
        for concept_id in concept_data:
            for rel in concept_data[concept_id]['relations']:
                concept_data[concept_id]['relations'][rel] = list(
                    concept_data[concept_id]['relations'][rel]
                )
        
        return concept_data


def demo_ontology_loading():
    """Demonstrate ontology loading"""
    print("=" * 70)
    print("ONTOLOGY INTEGRATION DEMO - Issue #13")
    print("=" * 70)
    print()
    
    integrator = OntologyIntegrator()
    
    # Test ConceptNet
    print("1. Loading from ConceptNet:")
    print("-" * 70)
    concept = "dog"
    triples = integrator.load_concept(concept, sources=['conceptnet'], limit_per_source=10)
    
    if triples:
        print(f"\nSample facts about '{concept}':")
        for i, triple in enumerate(triples[:5], 1):
            print(f"  {i}. {triple.subject} --[{triple.relation}]--> {triple.object}")
            print(f"     (confidence: {triple.confidence:.2f})")
    
    # Convert to MKU data
    print(f"\n2. Converting to MKU format:")
    print("-" * 70)
    mku_data = integrator.triples_to_mku_data(triples)
    
    for concept_id, data in list(mku_data.items())[:3]:
        print(f"\n  Concept: {concept_id}")
        print(f"    Source: {data['source']}")
        print(f"    Properties: {len(data['properties'])}")
        print(f"    Relations: {len(data['relations'])}")
        if data['relations']:
            rel_name = list(data['relations'].keys())[0]
            print(f"    Sample: {rel_name} → {data['relations'][rel_name][:2]}")
    
    print("\n" + "=" * 70)
    print("✓ Ontology integration demo complete!")
    print("=" * 70)


if __name__ == '__main__':
    demo_ontology_loading()
