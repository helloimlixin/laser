import os,time,concurrent.futures,json
from pathlib import Path
base=Path('/tmp/laser-matched680-vector6144-20261111/data')
for src,name in [('/workspace/Projects/data/imagenet/archives/ILSVRC2012_img_train.verified-20261005.tar','ILSVRC2012_img_train.tar'),('/workspace/Projects/data/imagenet/validation-verified-20261107/ILSVRC2012_img_val.tar','ILSVRC2012_img_val.tar')]:
 size=os.stat(src).st_size;dst=base/name;fd=os.open(src,os.O_RDONLY);out=os.open(dst,os.O_WRONLY|os.O_CREAT,0o644);os.ftruncate(out,size);start=time.monotonic();count=0
 def copy(offset):
  n=min(16*1024**2,size-offset);raw=os.pread(fd,n,offset);assert len(raw)==n
  assert os.pwrite(out,raw,offset)==n;return n
 with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
  for n in pool.map(copy,range(0,size,16*1024**2)):
   count+=n
   if count%(1024**3)==0:print(name,count/1024**3,'GiB',round(count/(time.monotonic()-start)/1e6,1),'MB/s',flush=True)
 os.close(fd);os.close(out)
 import hashlib
 with dst.open('rb') as stream: digest=hashlib.file_digest(stream,'md5').hexdigest()
 expected='1d675b47d978889d74fa0da5fadfb00e' if 'train' in name else '29b22e2961454d5413ddabcf34fc5622'
 assert digest==expected,(name,digest)
 (base/(name+'.verified.json')).write_text(json.dumps(dict(verified=True,md5=digest,bytes=size,source=src)))
 print('DONE',name,size,digest,flush=True)
