"""Index and locally cache the verified ImageNet training archive."""
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.training.pair_scratch_data import build_index

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--classes',type=Path,required=True)
    p.add_argument('--heldout',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--local-archive',type=Path)
    args=p.parse_args()
    print(json.dumps(build_index(args.source,args.classes,args.heldout,args.output,local_archive=args.local_archive)),flush=True)
