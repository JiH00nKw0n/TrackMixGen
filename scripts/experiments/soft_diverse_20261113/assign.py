from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess,json,hashlib
base=Path(__file__).resolve().parent
pilot=base.parent
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def work(label):
 task=label[:-2]; folder=base/label;folder.mkdir(exist_ok=True)
 retry=pilot/'duration_retries'/label/'attempt01'
 if (retry/'outcome.json').exists() and (r:=json.loads((retry/'outcome.json').read_text())).get('success') and r['steps']<r['original_steps']:
  p=retry/r['file']; assert digest(p)==r['sha256']
 else:
  orig=pilot/label; receipt=orig/'verified_result.json'
  if not receipt.exists():receipt=next(orig.rglob('verification.json'))
  meta=json.loads(receipt.read_text()); candidates=list(orig.rglob('*.hdf5'))
  if label=='rope01':p,=candidates
  else:p=next(p for p in candidates if digest(p)==meta['sha256'])
 meta=dict(task=task,scene=label,file=str(p),sha256=digest(p))
 (folder/'input.json').write_text(json.dumps(meta,indent=2))
 prompt=f"""You are the sole independent Astra annotator for {label}. Read {folder}/input.json for the selected verified HDF5. Compare the three released human samples in {base}/reference/{task}.hdf5. Determine the most source-faithful grasp completion boundary for this one sample, in ZERO-based PRE-ACTION frame indices, using both hands/object evidence and action timings (consult recording script/trace and images near transition if needed). Rope use official object_grasped exactly, including both actual finger joints (remember second obs sign) and nearest node <1.5cm; report first rising edge and any later drops (native flags will be calculated by root). Towel is manual in originals, true 5/7/7 frames after close begins in each source respectively; determine actual secure grasp in your own episode from the recorded states and find equivalent point, not a blind normalized time, validate following lift/transport. Inspect whether a failed grasp preceded the productive one; explain. Do not regenerate trajectories. Write {folder}/boundary.json with keys scene, boundary (integer zero-based first true), evidence (text), confidence, source_sha256. Only handle this episode. Root combines your boundary with calibrated official pose and nodal functions to actually annotate the HDF5. Work locally, CPU only; never modify input, other agent files or call another agent. Finish promptly."""
 with (folder/'agent.log').open('w') as log:
  proc=subprocess.run(['codex','exec','--ephemeral','-m','gpt-6-astra','-s','danger-full-access','-C',str(pilot.parents[1]),'-o',str(folder/'reply.txt'),prompt],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
 return label,proc.returncode,(folder/'boundary.json').exists()
if __name__=='__main__':
 labels=[t+f'{i:02}' for i in range(1,11) for t in ('rope','towel')]
 with ThreadPoolExecutor(max_workers=3) as pool:
  for item in pool.map(work,labels): print(item,flush=True)
