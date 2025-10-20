#!/usr/bin/env python3

import sys
import importlib
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Force reload modules
import models.ensemble.model_ensemble
importlib.reload(models.ensemble.model_ensemble)

from models.ensemble.model_ensemble import ModelEnsemble
import torch

# Test ModelEnsemble
print("Testing ModelEnsemble...")
m = ModelEnsemble([])

# Check methods
print("Available methods:")
for attr in dir(m):
    if 'predict' in attr:
        print(f"  {attr}")

# Test method existence
print(f"Has predict_weighted_average: {hasattr(m, 'predict_weighted_average')}")
print(f"Has predict_voting: {hasattr(m, 'predict_voting')}")

# Test method call
try:
    x = torch.randn(1, 10, 5)
    result = m.predict_weighted_average(x)
    print("✅ predict_weighted_average works!")
except Exception as e:
    print(f"❌ Error in predict_weighted_average: {e}")

try:
    x = torch.randn(1, 10, 5)
    result = m.predict_voting(x)
    print("✅ predict_voting works!")
except Exception as e:
    print(f"❌ Error in predict_voting: {e}")
