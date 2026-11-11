"""Check exact batch membership and the uneven final batch after repartitioning."""
import sys,json
from pathlib import Path
base=Path('/tmp/laser-beam16-duplicate-20261105')
sys.path.insert(0,str(base/'frozen/source'))
from src.training.exact_global_batch import ExactGlobalBatchSampler
from itertools import islice
class Dataset:
 def __len__(self):return 1281167

rows=[]
for epoch,step in [(17,108),(17,625),(18,0)]:
 original=ExactGlobalBatchSampler(Dataset(),2048,1,0,64,261001);original.set_epoch(epoch);original.set_start_batch(step*64)
 expected=[i for batch in islice(iter(original),64) for i in batch]
 combined=[]
 counts=[]
 for rank in range(2):
  s=ExactGlobalBatchSampler(Dataset(),2048,2,rank,16,261001);s.set_epoch(epoch);s.set_start_batch(step*16)
  indices=[i for batch in islice(iter(s),16) for i in batch];counts.append(len(indices));combined+=indices
 assert sorted(combined)==sorted(expected) and len(combined)==len(set(combined))
 rows.append(dict(epoch=epoch,update_in_epoch=step,global_count=len(combined),rank_counts=counts,exact_source_batch_membership=True))
report=dict(passed=True,batches=rows,evaluation_rank_counts=[25000,25000],evaluation_total=50000)
labels=[(i*2+rank)%1000 for rank in range(2) for i in range(25000)]
assert all(labels.count(c)==50 for c in range(1000))
p=Path('/workspace/Projects/laser/outputs/imnet-beam16-duplicate-20261105/layout-verification.json');p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
