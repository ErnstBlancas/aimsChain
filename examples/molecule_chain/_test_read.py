import sys, os
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src"))
from aimsChain.aimsio import read_aims

ini = read_aims("ini.in")
fin = read_aims("fin.in")
print(f"ini atoms: {len(ini.atoms)}")
print(f"fin atoms: {len(fin.atoms)}")
