#!/usr/bin/env python3
"""
Persistent Storage - Issue #31
SQLite-based persistence for knowledge graph

Phase 6: Production Readiness
Enables system to save/load state across sessions
"""

import sqlite3
import json
import pickle
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    from .mln import KnowledgeGraph, MonadicKnowledgeUnit, InferenceRule
except ImportError:
    from mln import KnowledgeGraph, MonadicKnowledgeUnit, InferenceRule


class PersistentKnowledgeGraph:
    """
    Persistent knowledge graph with SQLite backend
    
    Features:
    - Save/load concepts and relations
    - Transaction support
    - Query history
    - Incremental updates
    """
    
    def __init__(self, db_path: str = "mln_knowledge.db", kg: Optional[KnowledgeGraph] = None):
        """
        Args:
            db_path: Path to SQLite database
            kg: Existing KnowledgeGraph to persist (optional)
        """
        self.db_path = Path(db_path)
        self.kg = kg or KnowledgeGraph(use_gpu=False)
        self.conn = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Create database schema if it doesn't exist"""
        self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()
        
        # Concepts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS concepts (
                concept_id TEXT PRIMARY KEY,
                deep_structure TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Relations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES concepts(concept_id),
                FOREIGN KEY (target_id) REFERENCES concepts(concept_id),
                UNIQUE(source_id, relation_type, target_id)
            )
        ''')
        
        # Inference rules table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inference_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_data BLOB NOT NULL,
                priority INTEGER DEFAULT 10,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Query history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                result TEXT,
                execution_time REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
        
        # Set initial metadata
        cursor.execute('''
            INSERT OR IGNORE INTO metadata (key, value) 
            VALUES ('version', '1.0.0')
        ''')
        cursor.execute('''
            INSERT OR IGNORE INTO metadata (key, value) 
            VALUES ('created_at', datetime('now'))
        ''')
        self.conn.commit()
    
    def save(self):
        """Save current knowledge graph to database"""
        cursor = self.conn.cursor()
        
        # Save concepts
        for concept_id, mku in self.kg.nodes.items():
            deep_structure_json = json.dumps(mku.deep_structure)
            
            cursor.execute('''
                INSERT OR REPLACE INTO concepts (concept_id, deep_structure, updated_at)
                VALUES (?, ?, datetime('now'))
            ''', (concept_id, deep_structure_json))
            
            # Save relations
            for relation_type, related_ids in mku.relations.items():
                for target_id in related_ids:
                    cursor.execute('''
                        INSERT OR IGNORE INTO relations (source_id, relation_type, target_id)
                        VALUES (?, ?, ?)
                    ''', (concept_id, relation_type, target_id))
        
        # Save inference rules (using pickle for simplicity)
        cursor.execute('DELETE FROM inference_rules')  # Clear old rules
        for rule in self.kg.inference_rules:
            rule_blob = pickle.dumps(rule)
            cursor.execute('''
                INSERT INTO inference_rules (rule_data, priority)
                VALUES (?, ?)
            ''', (rule_blob, rule.priority))
        
        # Update metadata
        cursor.execute('''
            UPDATE metadata 
            SET value = ?, updated_at = datetime('now')
            WHERE key = 'last_save'
        ''', (str(len(self.kg.nodes)),))
        
        if cursor.rowcount == 0:
            cursor.execute('''
                INSERT INTO metadata (key, value)
                VALUES ('last_save', ?)
            ''', (str(len(self.kg.nodes)),))
        
        self.conn.commit()
        return len(self.kg.nodes)
    
    def load(self) -> KnowledgeGraph:
        """Load knowledge graph from database"""
        cursor = self.conn.cursor()
        
        # Clear current graph
        self.kg = KnowledgeGraph(use_gpu=False)
        
        # Load concepts
        cursor.execute('SELECT concept_id, deep_structure FROM concepts')
        concepts_data = cursor.fetchall()
        
        for concept_id, deep_structure_json in concepts_data:
            deep_structure = json.loads(deep_structure_json)
            mku = MonadicKnowledgeUnit(
                concept_id=concept_id,
                deep_structure=deep_structure
            )
            # Add without reflecting (we'll restore relations separately)
            self.kg.nodes[concept_id] = mku
        
        # Load relations
        cursor.execute('SELECT source_id, relation_type, target_id FROM relations')
        relations_data = cursor.fetchall()
        
        for source_id, relation_type, target_id in relations_data:
            if source_id in self.kg.nodes:
                mku = self.kg.nodes[source_id]
                if relation_type not in mku.relations:
                    mku.relations[relation_type] = set()
                mku.relations[relation_type].add(target_id)
        
        # Load inference rules
        try:
            cursor.execute('SELECT rule_data FROM inference_rules ORDER BY priority')
            rules_data = cursor.fetchall()
            
            for (rule_blob,) in rules_data:
                rule = pickle.loads(rule_blob)
                self.kg.inference_rules.append(rule)
        except Exception as e:
            print(f"Warning: Could not load inference rules: {e}")
        
        return self.kg
    
    def log_query(self, query: str, result: Any, execution_time: float):
        """Log a query for history tracking"""
        cursor = self.conn.cursor()
        result_str = str(result)[:1000]  # Limit result size
        
        cursor.execute('''
            INSERT INTO query_history (query, result, execution_time)
            VALUES (?, ?, ?)
        ''', (query, result_str, execution_time))
        
        self.conn.commit()
    
    def get_query_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent query history"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT query, result, execution_time, timestamp
            FROM query_history
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        return [
            {
                'query': row[0],
                'result': row[1],
                'execution_time': row[2],
                'timestamp': row[3]
            }
            for row in cursor.fetchall()
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM concepts')
        concept_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM relations')
        relation_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM inference_rules')
        rule_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM query_history')
        query_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT value FROM metadata WHERE key = "created_at"')
        created_at = cursor.fetchone()
        created_at = created_at[0] if created_at else "Unknown"
        
        cursor.execute('SELECT value FROM metadata WHERE key = "last_save"')
        last_save = cursor.fetchone()
        last_save = last_save[0] if last_save else "Never"
        
        return {
            'database': str(self.db_path),
            'concepts': concept_count,
            'relations': relation_count,
            'inference_rules': rule_count,
            'queries_logged': query_count,
            'created_at': created_at,
            'last_save': last_save
        }
    
    def export_json(self, output_path: str):
        """Export knowledge graph to JSON"""
        export_data = {
            'concepts': {},
            'relations': [],
            'metadata': self.get_statistics()
        }
        
        # Export concepts
        for concept_id, mku in self.kg.nodes.items():
            export_data['concepts'][concept_id] = {
                'deep_structure': mku.deep_structure,
                'relations': {
                    rel_type: list(targets)
                    for rel_type, targets in mku.relations.items()
                }
            }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def import_json(self, input_path: str):
        """Import knowledge graph from JSON"""
        with open(input_path, 'r') as f:
            import_data = json.load(f)
        
        # Import concepts
        for concept_id, concept_data in import_data['concepts'].items():
            mku = MonadicKnowledgeUnit(
                concept_id=concept_id,
                deep_structure=concept_data['deep_structure']
            )
            # Restore relations
            for rel_type, targets in concept_data.get('relations', {}).items():
                mku.relations[rel_type] = set(targets)
            
            self.kg.nodes[concept_id] = mku
        
        # Save to database
        self.save()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def demo():
    """Demonstrate persistent storage"""
    print("=" * 70)
    print("PERSISTENT STORAGE - Issue #31 Demo")
    print("SQLite-based Knowledge Graph Persistence")
    print("=" * 70)
    
    # Create temporary database
    db_path = "demo_mln.db"
    
    print("\n1. CREATE AND POPULATE KNOWLEDGE GRAPH")
    print("-" * 70)
    
    with PersistentKnowledgeGraph(db_path) as pkg:
        # Add concepts
        from mln import MonadicKnowledgeUnit
        
        concepts = {
            'animal': {'type': 'organism', 'breathes': True},
            'mammal': {'type': 'animal', 'warm_blooded': True},
            'dog': {'type': 'mammal', 'domesticated': True},
        }
        
        for concept_id, properties in concepts.items():
            mku = MonadicKnowledgeUnit(
                concept_id=concept_id,
                deep_structure={'properties': properties}
            )
            pkg.kg.add_concept(mku)
        
        print(f"Added {len(concepts)} concepts")
        
        # Save to database
        saved_count = pkg.save()
        print(f"Saved {saved_count} concepts to database")
        
        # Show statistics
        stats = pkg.get_statistics()
        print(f"\nDatabase Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    print("\n2. LOAD FROM DATABASE")
    print("-" * 70)
    
    with PersistentKnowledgeGraph(db_path) as pkg:
        # Load from database
        kg = pkg.load()
        print(f"Loaded {len(kg.nodes)} concepts from database")
        
        for concept_id in kg.nodes:
            print(f"  - {concept_id}")
    
    print("\n3. EXPORT TO JSON")
    print("-" * 70)
    
    with PersistentKnowledgeGraph(db_path) as pkg:
        pkg.load()
        json_path = "demo_export.json"
        pkg.export_json(json_path)
        print(f"Exported to {json_path}")
    
    print("\n" + "=" * 70)
    print("KEY CAPABILITY: Knowledge persists across sessions!")
    print("=" * 70)
    
    # Cleanup
    import os
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists("demo_export.json"):
        os.remove("demo_export.json")


if __name__ == "__main__":
    demo()
