"""
MSME-RL Environment.

A mixed portfolio of 20 MSME + 10 startup accounts across a 36-month loan cycle.
Teaches a 1.7B LM to decode two opposite linguistic strategies from reward signal alone:
  - MSME owners who UNDERSTATE problems (Hindi/Hinglish/Marathi)
  - Startup founders who OVERSTATE health (pitch-deck English)

Calibrated to: RBI FY24, NASSCOM/CIBIL 2023, SIDBI MSME Pulse, IBA 2023.
"""

from .client import MSMERLEnv
from .models import MSMERLAction, MSMERLObservation

__all__ = [
    "MSMERLAction",
    "MSMERLObservation",
    "MSMERLEnv",
]