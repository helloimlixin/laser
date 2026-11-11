"""Run the existing verified uploader against the explicitly selected run folder."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(args.base/'runtime'))
    spec=importlib.util.spec_from_file_location('scoped_cloud_sync',args.base/'runtime/cloud_worker.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.OUT=args.output;module.main()
