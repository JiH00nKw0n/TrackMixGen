"""Wait for all individual decisions, assemble each source, and verify its remote bytes."""
import hashlib,json,subprocess,time
from pathlib import Path
from build import build,BASE
REMOTE='/home/gpuadmin/workspace/jihoonkwon/soft_source_demo_pilot_20261112/annotation_20261113'

def run(*args): return subprocess.check_output(args,text=True).strip()
def publish(task):
    build(task)
    src=BASE/(task+'.hdf5')
    with src.open('rb') as f:
        sha=hashlib.file_digest(f,'sha256').hexdigest()
    with src.open('rb') as f:
        md5=hashlib.file_digest(f,'md5').hexdigest()
    d=f'{REMOTE}/data/sources/softmimicgen/{task}'
    run('scp','-q',str(src),f'isaac:{d}/source.hdf5.partial')
    assert run('ssh','isaac','sha256sum',d+'/source.hdf5.partial').split()[0]==sha
    run('ssh','isaac','mv',d+'/source.hdf5.partial',d+'/source.hdf5')
    manifest={'path':f'/data/sources/softmimicgen/{task}/source.hdf5','md5':md5,'sha256':sha,'bytes':src.stat().st_size,'demos':10,'expected_sources':10}
    mf=BASE/(task+'_source.manifest.json');mf.write_text(json.dumps(manifest,indent=2))
    run('scp','-q',str(mf),f'isaac:{d}/source.manifest.json')
    run('scp','-q',str(BASE/(task+'_annotations.json')),f'isaac:{REMOTE}/data/')
    print(task,sha,flush=True)
pending={'rope','towel'}
while pending:
    for task in sorted(pending):
        if len(list(BASE.glob(task+'[0-9][0-9]/boundary.json')))==10:
            publish(task)
            pending.remove(task)
    if pending:time.sleep(5)
