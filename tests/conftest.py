import sys
import os

# Add src to path so lib imports work without _init_paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
