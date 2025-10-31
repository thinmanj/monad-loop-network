#!/usr/bin/env python3
"""
MLN Demo: Self-Referential Knowledge System
Demonstrates GEB strange loops + Chomsky deep structure + Leibniz monads
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mln import demo

if __name__ == '__main__':
    demo()
