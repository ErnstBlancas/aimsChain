import sys, os
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src"))
from aimsChain.config import Control

control = Control()
print("periodic_interp:", hasattr(control, "periodic_interp"))
print("periodic_interpolation:", hasattr(control, "periodic_interpolation"))
for attr in sorted(dir(control)):
    if "periodic" in attr or "interp" in attr:
        print(f"  {attr} = {getattr(control, attr)}")
