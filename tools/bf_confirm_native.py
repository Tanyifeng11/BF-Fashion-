"""BF-Fashion原生条件对齐工具。只补缺图；确认来源后才允许generate，不训练。"""
import argparse,contextlib,hashlib,json,os,platform,random,subprocess,sys,time,tarfile,io,shutil
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def version(repo):return subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
def prepare(args):
 from PIL import Image
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 archive=Path(args.confirm_archive);assert sha(archive)==args.archive_sha256
 records=[]
 with tarfile.open(archive,'r:gz') as tar:
  manifest=json.loads(tar.extractfile('protocol/input_manifest.json').read().decode('utf-8'))
  assert len(manifest)==96
  for row in manifest:
   i=row['id'];assert i.isdigit();item=dict(id=i,caption_M12=row['caption'],BF_text_used=False)
   for kind in ['sketch','reference','gt']:
    name=f'inputs/{i}/{kind}_canonical.png';blob=tar.extractfile(name).read();image=Image.open(io.BytesIO(blob)).convert('RGB')
    digest=hashlib.sha256(image.tobytes()).hexdigest();assert digest==row[kind+'_rgb_sha256']
    target=out/'common_inputs'/i/(kind+'_canonical.png');target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert target.read_bytes()==blob
    else:target.write_bytes(blob)
    item[kind]=str(target.resolve());item[kind+'_rgb_sha256']=digest;item[kind+'_png_sha256']=sha(target);item[kind+'_original_JPEG_sha256']=row[kind+'_sha256']
   records.append(item)
 write(out/'BF_confirm_manifest.json',records)
 write(out/'BF_prepare.json',dict(source_archive=str(archive.resolve()),source_archive_sha256=args.archive_sha256,manifest_sha256=sha(out/'BF_confirm_manifest.json'),identities=96,canonical_RGB_checked=288,new_generation=0,new_training=0))
 print('BF_INPUTS_PREPARED',96,flush=True)
def audit(args):
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 import torch
 from omegaconf import OmegaConf
 checkpoint=Path(args.ckpt).resolve();vq=Path(args.vq).resolve();repo=Path(args.repo).resolve()
 config=OmegaConf.load(repo/'configs/latent-diffusion/mydata_ldm-vq-f8_test.yaml')
 assert config.model.params.scale_1==1.1 and config.model.params.scale_2==1.5
 state=torch.load(checkpoint,map_location='cpu');assert 'state_dict' in state
 training_config=Path(args.training_config).resolve();saved=OmegaConf.load(training_config)
 saved_metadata=OmegaConf.select(saved,'data.params.train.params.metadata_path')
 source_repo=Path(args.source_repo).resolve();source_files=['dataset.py','ldm/models/diffusion/ddpm.py','ldm/models/diffusion/ddim.py','ldm/modules/diffusionmodules/Fusion.py','ldm/modules/diffusionmodules/openaimodel.py','ldm/models/autoencoder.py']
 source_hashes={p:dict(source=sha(source_repo/p),runner=sha(repo/p)) for p in source_files}
 assert all(v['source']==v['runner'] for v in source_hashes.values()),'实际旧模型源码与生成工作树不一致'
 snapshots=out/'audit_sources';snapshots.mkdir(exist_ok=True)
 shutil.copy2(training_config,snapshots/'saved_training_project.yaml');shutil.copy2(args.train_manifest,snapshots/'train_manifest.json')
 target=read(args.confirm_manifest);ids=[r['id'] for r in target];assert len(ids)==len(set(ids))==96
 train=read(args.train_manifest);train_ids={Path(r['cloth']).stem for r in train}
 overlap=sorted(set(ids)&train_ids)
 # 这里只登记；audit完成不等于研究者已经核验checkpoint实际对应此训练清单。
 v=dict(checkpoint_path=str(checkpoint),checkpoint_sha256=sha(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
  epoch=state.get('epoch'),global_step=state.get('global_step'),weights_keys=len(state['state_dict']),checkpoint_top_level_keys=sorted(state),
  state_dict_elements=sum(v.numel() for v in state['state_dict'].values() if hasattr(v,'numel')),
  training_config_path=str(training_config),training_config_sha256=sha(training_config),saved_training_metadata_path=saved_metadata,
  saved_training_config=OmegaConf.to_container(saved,resolve=True),
  vq_path=str(vq),vq_sha256=sha(vq),repo=str(repo),commit=version(repo),git_diff=subprocess.check_output(['git','diff','--stat'],cwd=repo,text=True),
  source_repo=str(source_repo),source_commit=version(source_repo),source_files_sha256=source_hashes,
  python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,platform=platform.platform(),
  train_manifest_path=str(Path(args.train_manifest).resolve()),train_manifest_sha256=sha(args.train_manifest),train_manifest_mtime_ns=Path(args.train_manifest).stat().st_mtime_ns,checkpoint_mtime_ns=checkpoint.stat().st_mtime_ns,train_count=len(train),train_test_identity_overlap=overlap,
  confirm_manifest_sha256=sha(args.confirm_manifest),native_config=OmegaConf.to_container(config,resolve=True),text_supported=False,
  comparison_status='native settings, condition difference: M12 uses identity-specific caption',
  seed_protocol=[42,43,44],native_size=[256,256],evaluation_size=[384,512],ddim_steps=200,eta=0,
  output_geometry='directBILINEAR384x512 from native256x256, no crop/background/color changes',
  checkpoint_training_provenance_approved=False,new_training=0)
 write(out/'BF_checkpoint_audit.json',v)
 print(json.dumps({k:v[k] for k in ['checkpoint_sha256','epoch','global_step','commit','train_count','train_test_identity_overlap']},ensure_ascii=False),flush=True)

def generate(args):
 import numpy as np
 import torch
 from PIL import Image
 from omegaconf import OmegaConf
 repo=Path(args.repo).resolve();sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'src/taming-transformers'))
 from ldm.util import instantiate_from_config
 from ldm.models.diffusion.ddim import DDIMSampler
 out=Path(args.out);auditinfo=read(out/'BF_checkpoint_audit.json');approval=read(args.approval)
 assert approval['checkpoint_training_provenance_approved'] and approval['checkpoint_sha256']==auditinfo['checkpoint_sha256']
 assert not auditinfo['train_test_identity_overlap'] and sha(args.ckpt)==auditinfo['checkpoint_sha256'] and sha(args.vq)==auditinfo['vq_sha256']
 rows=read(args.confirm_manifest);assert sha(args.confirm_manifest)==auditinfo['confirm_manifest_sha256']
 config=OmegaConf.load(repo/'configs/latent-diffusion/mydata_ldm-vq-f8_test.yaml')
 config.model.params.first_stage_config.params.ckpt_path=str(Path(args.vq).resolve())
 model=instantiate_from_config(config.model);weights=torch.load(args.ckpt,map_location='cpu')
 missing,unexpected=model.load_state_dict(weights['state_dict'],strict=False)
 loading=dict(missing_keys=list(missing),unexpected_keys=list(unexpected),dtype='nativefloat32',EMA_context=False,model_parameters=sum(p.numel() for p in model.parameters()))
 write(out/('BF_loading_'+os.environ.get('SLURM_JOB_ID','local')+'.json'),loading);write(out/'BF_loading.json',loading)
 assert not any(k.startswith(('model.diffusion_model.','cond_stage_model.','first_stage_model.')) for k in missing),'关键推理权重缺失，停止'
 del weights
 model.cuda().eval().requires_grad_(False);sampler=DDIMSampler(model);counts=[]
 # 官方DDIM会将中间latent写入cwd；隔离保存，不覆盖项目文件。
 runtime=out/'native_runtime_artifacts';runtime.mkdir(exist_ok=True);os.chdir(runtime)
 torch.backends.cudnn.benchmark=False
 def tensor(im):
  # 与原dataset完全一致：uint8先除127.5，再转float32。
  value=(np.asarray(im).astype('uint8')/127.5-1.0).astype('float32')
  if value.ndim==2:value=value[:,:,None]
  return torch.from_numpy(value.transpose(2,0,1)[None]).cuda()
 def run(row,seed):
  sketch=Image.open(row['sketch']).convert('L');reference=Image.open(row['reference']).convert('RGB')
  assert sketch.size==reference.size==(256,256)
  # 图像哈希必须是冻结canonical版本；生成阶段完全不读取GT。
  for p,key in [(row['sketch'],'sketch_rgb_sha256'),(row['reference'],'reference_rgb_sha256')]:
   assert hashlib.sha256(Image.open(p).convert('RGB').tobytes()).hexdigest()==row[key]
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  st=tensor(sketch);st=st if st.shape[1]==1 else st[:,:1];ref=tensor(reference)
  cond={'cond':model.get_learned_conditioning({'sketch':st,'texture':ref})}
  uncond={'two_zero':model.get_learned_conditioning({'sketch':torch.zeros_like(st),'texture':torch.zeros_like(ref)}),
          'one_zero':model.get_learned_conditioning({'sketch':torch.zeros_like(st),'texture':ref})}
  latent=torch.randn((1,4,32,32),device='cuda');initial=hashlib.sha256(latent.detach().cpu().numpy().tobytes()).hexdigest()
  torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.perf_counter()
  samples,_=sampler.sample(S=200,conditioning=cond,batch_size=1,shape=[4,32,32],verbose=False,
   unconditional_guidance_scale={'scale_1':1.1,'scale_2':1.5},unconditional_conditioning=uncond,eta=0.,x_T=latent)
  decoded=torch.clamp((model.decode_first_stage(samples)+1)/2,min=0,max=1)
  torch.cuda.synchronize();seconds=time.perf_counter()-begin
  # torchvision save_image默认uint8舍入口径，固定为PNG不再JPEG压缩。
  array=decoded[0].detach().permute(1,2,0).mul(255).add(.5).clamp(0,255).to(torch.uint8).cpu().numpy()
  im=Image.fromarray(array);meta=dict(id=row['id'],seed=seed,checkpoint_sha256=auditinfo['checkpoint_sha256'],code_commit=version(repo),
    sketch_rgb_sha256=row['sketch_rgb_sha256'],reference_rgb_sha256=row['reference_rgb_sha256'],sketch_file_sha256=sha(row['sketch']),reference_file_sha256=sha(row['reference']),
    text_used=False,GT_used_in_inference=False,initial_latent_sha256=initial,DDIM_steps=200,eta=0.,scale_1=1.1,scale_2=1.5,
    native_size=[256,256],evaluation_size=[384,512],seconds_native_inference=seconds,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,gpu=torch.cuda.get_device_name(),
    timing_scope='conditioning preparation excluded; sampler and VQ decode included; PNG/resize excluded',scientific_training_updates=0,EMA_context=False)
  return im,meta
 with torch.no_grad():
  for row in rows:
   for seed in [42,43,44]:
    native=out/'native_png'/row['id']/f'seed{seed}.png';dest=out/'outputs/BF-Fashion'/row['id']/f'seed{seed}.png';meta=dest.with_suffix('.json')
    if native.exists() or dest.exists() or meta.exists():
     assert native.exists() and dest.exists() and meta.exists(),'部分产物需先审计，不自动覆盖'
     old=read(meta);assert old['id']==row['id'] and old['seed']==seed and old['checkpoint_sha256']==auditinfo['checkpoint_sha256'] and old['png_sha256']==sha(dest) and old['native_png_sha256']==sha(native)
     assert old['sketch_rgb_sha256']==row['sketch_rgb_sha256'] and old['reference_rgb_sha256']==row['reference_rgb_sha256']
     counts.append(dict(id=row['id'],seed=seed,reused=True));continue
    image,details=run(row,seed);native.parent.mkdir(parents=True,exist_ok=True);dest.parent.mkdir(parents=True,exist_ok=True)
    image.save(native);image.resize((384,512),Image.BILINEAR).save(dest)
    details.update(native_png_sha256=sha(native),png_sha256=sha(dest),success=True);write(meta,details);counts.append(dict(id=row['id'],seed=seed,reused=False))
    print('BF_OUTPUT',row['id'],seed,len(counts),flush=True)
    if args.limit and len(counts)>=args.limit:
     write(out/'BF_smoke_complete.json',dict(images=len(counts),checkpoint_sha256=auditinfo['checkpoint_sha256'],records=counts,new_training=0,formal_outputs_reused_by_full_run=True));return
 write(out/'BF_generation_complete.json',dict(images=len(counts),checkpoint_sha256=auditinfo['checkpoint_sha256'],records=counts,new_training=0));assert len(counts)==288

def approve(args):
 out=Path(args.out);a=read(out/'BF_checkpoint_audit.json')
 assert a['checkpoint_sha256']==args.expected_checkpoint_sha256 and a['train_manifest_sha256']==args.expected_train_sha256
 assert (a['epoch'],a['global_step'],a['train_count'])==(26,49166,45384) and not a['train_test_identity_overlap']
 assert a['saved_training_metadata_path']==a['train_manifest_path'] and a['train_manifest_mtime_ns']<a['checkpoint_mtime_ns']
 assert all(v['source']==v['runner'] for v in a['source_files_sha256'].values())
 write(out/'BF_provenance_approval.json',dict(checkpoint_training_provenance_approved=True,checkpoint_sha256=a['checkpoint_sha256'],train_manifest_sha256=a['train_manifest_sha256'],evidence='checkpoint folder and saved project config identify the same training metadata; current manifest predates checkpoint;45384 identities checked,96 confirmation identities excluded; source network files match',limitation='No immutable manifest fingerprint saved at historical training time; current contents and timestamps support reconstruction, not proof of no pretraining exposure or a fresh blind test.',new_training=0))
 print('BF_PROVENANCE_RECONSTRUCTION_ACCEPTED_WITH_DISCLOSED_LIMITS',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','audit','approve','generate']);p.add_argument('--repo');p.add_argument('--source-repo');p.add_argument('--ckpt');p.add_argument('--vq');p.add_argument('--out',required=True);p.add_argument('--confirm-manifest');p.add_argument('--confirm-archive');p.add_argument('--archive-sha256');p.add_argument('--train-manifest');p.add_argument('--training-config');p.add_argument('--approval');p.add_argument('--limit',type=int,default=0);p.add_argument('--expected-checkpoint-sha256');p.add_argument('--expected-train-sha256');args=p.parse_args()
 globals()[args.action](args)
