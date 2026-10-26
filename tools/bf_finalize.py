"""所有本轮实际作业终态后保留成本/失败与可独立验算归档，不包含模型权重。"""
import hashlib,json,re,subprocess,tarfile
from pathlib import Path
R=Path('/share/home/u2515283058/Mymodel/output_eval/m12_external_closeout_20261026')
repo=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
assert read(R/'BF_generation_complete.json')['images']==288
assert read(R/'metrics_BF/complete.json')['images']==288
assert read(R/'metrics_BF/M12_GPU_replay_verification.json')['passed']
ids=[]
for p in sorted((R/'resources').glob('*_job_ids.txt')):
 for line in p.read_text().splitlines():
  if line.strip():
   assert re.fullmatch(r'\d+(;\S+)?',line.strip()),str(p)
   ids.append(int(line.split(';')[0]))
assert ids and len(ids)==len(set(ids)), '作业登记缺失或重复；不能漏掉失败/重试'
fields=['JobIDRaw','State','ExitCode','ElapsedRaw','AllocCPUS','AllocTRES','NodeList']
raw=subprocess.check_output(['sacct','-j',','.join(map(str,ids)),'--format='+','.join(fields),'--parsable2','--noheader'],text=True)
(R/'resources/final_sacct.psv').write_text(raw)
records=[]
for line in raw.splitlines():
 values=line.split('|');row=dict(zip(fields,values));job=row['JobIDRaw']
 if job not in {str(i) for i in ids}:continue
 assert row['State'].split()[0] in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'],row
 g=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',row['AllocTRES']);gpu=int(g.group(1)) if g else 0
 row.update(gpus=gpu,seconds=int(row['ElapsedRaw']),GPU_hours=int(row['ElapsedRaw'])*gpu/3600);records.append(row)
assert {int(r['JobIDRaw']) for r in records}==set(ids)
write(R/'resources/final_accounting.json',dict(jobs=records,actual_job_ids=ids,all_actual_jobs_terminal=True,GPU_hours=sum(r['GPU_hours'] for r in records),failed_job_ids=[r['JobIDRaw'] for r in records if r['State']!='COMPLETED'],rejected_submissions='preserve command logs; no rejected allocation may be counted as GPU time',new_training=0,other_experiments_untouched=True))
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
subprocess.run(['git','archive','--output',str(R/'BF_code_sources.tar'),'HEAD'],cwd=repo,check=True)
paths=[]
for folder in ['outputs','native_png','common_inputs','audit_sources','resources','metrics_BF']:
 paths.extend(p for p in (R/folder).rglob('*') if p.is_file())
paths.extend(R.glob('BF_*.json'));paths.append(R/'BF_code_sources.tar')
paths=sorted(set(paths));manifest={p.relative_to(R).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in paths}
write(R/'portable_manifest.json',dict(files=manifest,code_commit=commit,raw_images=288,native_images=288,generated_with_GT=False,report_uploaded=False,human_masks='NOT_RUN_AT_SERVER_PACKAGING'))
archive=R/'bf_external_review.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for p in paths+[R/'portable_manifest.json']:tar.add(p,arcname=p.relative_to(R).as_posix(),recursive=False)
(R/'bf_external_review.tar.gz.sha256').write_text(sha(archive)+'  bf_external_review.tar.gz\n')
print('BF_SERVER_READY',sha(archive),'jobs',ids,flush=True)
