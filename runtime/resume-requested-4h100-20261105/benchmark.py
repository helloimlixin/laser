import os,sys,json,time,gc
from pathlib import Path
b=Path('/tmp/laser-resume-h100-20261105/frozen')
gpu=int(sys.argv[1]);micro=int(sys.argv[2]);sdpa=sys.argv[3]
os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),LASER_FUSED_ADAMW='1',LASER_COMPILE_NATIVE_LOSS='1',LASER_TRAIN_SDPA=sdpa,TORCHINDUCTOR_COMPILE_THREADS='2',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTORCH_ALLOC_CONF='expandable_segments:True')
sys.path[:0]=[str(b),str(b/'source')]
import train
import torch
from src.training import rq_imagenet_shared as s
from src.training.laser_joint_recipe import JointLaserRecipe
from runtime.training_acceleration import configure_optimizer
from runtime.checkpoint_memory import detach_cpu_optimizer_state
out=b.parent/f'benchmark-{gpu}.json'
try:
 torch.set_num_threads(4);torch.cuda.set_device(0);device=torch.device('cuda',0)
 s.BASE=b;s.PROFILE=JointLaserRecipe(b)
 aux,_=s.PROFILE.tokenizer(device)
 model,opt,config=s.fresh_model(device)
 p=torch.load(b/'inputs/resume-checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)
 model.load_state_dict(p['state_dict']);opt.load_state_dict(p['optimizer']);configure_optimizer(opt)
 detach_cpu_optimizer_state(opt);del p;gc.collect()
 model.train();scaler=torch.amp.GradScaler('cuda',enabled=False)
 images=torch.rand(micro,3,256,256,device=device)*2-1
 labels=torch.arange(micro,device=device)%1000
 times=[];losses=[]
 for i in range(6):
  torch.cuda.synchronize();start=time.monotonic()
  loss=s.training_backward(s.PROFILE,model,aux,images,labels,scaler,1)
  if i==0:opt.step()
  opt.zero_grad(set_to_none=True);torch.cuda.synchronize()
  times.append(time.monotonic()-start);losses.append(float(loss))
 report=dict(passed=True,microbatch=micro,sdpa=sdpa,seconds=times,losses=losses,
     median_images_per_second=micro/float(torch.tensor(times[2:]).median()),
     peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
except Exception as e:
 report=dict(passed=False,microbatch=micro,sdpa=sdpa,error=str(e))
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
