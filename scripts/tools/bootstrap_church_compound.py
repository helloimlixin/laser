#!/usr/bin/env python3
"""Download a pinned private W&B runtime using only Python's standard library."""
import base64
import hashlib
import json
import netrc
import os
from pathlib import Path
import subprocess
import urllib.parse
import urllib.request


def main():
    local = Path(os.environ['CHURCH_LOCAL_ROOT'])
    local.mkdir(parents=True, exist_ok=True)
    destination = local / 'bundle'
    expected = os.environ['CHURCH_BUNDLE_SHA256']
    marker = destination / '.bundle-verified'
    if marker.is_file() and marker.read_text().strip() == expected:
        return
    key = os.environ.get('WANDB_API_KEY')
    if not key:
        credentials = netrc.netrc().authenticators('api.wandb.ai')
        key = None if credentials is None else credentials[2]
    if not key:
        raise RuntimeError('W&B credentials unavailable for private runtime staging')
    auth = 'Basic ' + base64.b64encode(('api:' + key).encode()).decode()
    query = '''query($entity:String!,$project:String!,$run:String!,$names:[String]) {
      project(name:$project,entityName:$entity) {
        run(name:$run) { files(names:$names,first:1) {
          edges { node { name url(upload:false) sizeBytes md5 } }
        } }
      }
    }'''
    variables = dict(entity='helloimlixin-rutgers',project='laser',
        run='church-compound-amarel-runtime-20260919',names=[os.environ['CHURCH_BUNDLE_FILE']])
    request = urllib.request.Request('https://api.wandb.ai/graphql',
        data=json.dumps(dict(query=query,variables=variables)).encode(),
        headers={'Authorization':auth,'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    if result.get('errors'):
        raise RuntimeError('Unable to resolve private runtime file')
    rows = result['data']['project']['run']['files']['edges']
    if len(rows) != 1 or rows[0]['node']['name'] != os.environ['CHURCH_BUNDLE_FILE']:
        raise RuntimeError('Pinned runtime file was not found')
    row = rows[0]['node']

    class SafeRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
            if redirected is not None and urllib.parse.urlsplit(req.full_url).netloc != urllib.parse.urlsplit(newurl).netloc:
                redirected.remove_header('Authorization')
            return redirected

    request = urllib.request.Request(row['url'])
    if urllib.parse.urlsplit(row['url']).hostname == 'api.wandb.ai':
        request.add_header('Authorization',auth)
    archive = local / 'runtime-download.tar'
    digest = hashlib.sha256()
    print(json.dumps(dict(phase='downloading_runtime',host=os.uname().nodename,bytes=int(row['sizeBytes']))),flush=True)
    with urllib.request.build_opener(SafeRedirect()).open(request,timeout=120) as response, archive.open('wb') as output:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            output.write(chunk)
    if digest.hexdigest() != expected or archive.stat().st_size != int(row['sizeBytes']):
        raise RuntimeError('Runtime package hash/size mismatch')
    destination.mkdir(exist_ok=True)
    # The archive was created by this task and authenticated by its pinned hash.
    subprocess.run(['tar','-xf',str(archive),'-C',str(destination)],check=True)
    marker.write_text(expected+'\n')
    archive.unlink()
    print(json.dumps(dict(phase='runtime_verified',host=os.uname().nodename,sha256=expected)),flush=True)


if __name__ == '__main__':
    main()
