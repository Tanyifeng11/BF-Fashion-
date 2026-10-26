"""冻结PNG后先复现旧M12 GPU评价，再评BF；不覆盖任何历史分数。"""
import ast,hashlib,json,math,sys
from pathlib import Path
import cv2,numpy as np,torch,lpips
from PIL import Image
R=Path('/share/home/u2515283058/Mymodel/output_eval/m12_external_closeout_20261026')
OLD=R.parent/'m12_gain_independent_confirm_20261020'
SIZE=(384,512);SEEDS=[42,43,44]
read=lambda p:json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
frozen=Path(__file__).parent/'bf_eval_frozen'
assert sha(frozen/'garment_mask_utils.py')=='e7fb08d1838ef817757eca13c2bff92ca3e9eaccc673569c0513a97e78d73946'
assert sha(frozen/'e44_metrics_frozen.py')=='d409afbe80c0aa21fb26bf638fdb083b633d96738ea8f5310e2835d1daa29b3a'
sys.path.insert(0,str(frozen))
from garment_mask_utils import build_sketch_garment_mask,estimate_cloth_foreground_mask,mask_image_to_bool
ns={'np':np,'cv2':cv2}
tree=ast.parse((frozen/'e44_metrics_frozen.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['external_contour','contour_f1']],type_ignores=[]),'original_contours','exec'),ns)
out=R/'metrics_BF';out.mkdir(exist_ok=True)
rows=read(R/'BF_confirm_manifest.json');assert len({r['id'] for r in rows})==96
old=read(OLD/'metrics/per_image.json');index={(r['variant'],r['id'],r['seed']):r for r in old}
complete=read(R/'BF_generation_complete.json');assert complete['images']==288
images=[]
for row in rows:
 for seed in SEEDS:
  p=R/'outputs/BF-Fashion'/row['id']/('seed%d.png'%seed);m=read(p.with_suffix('.json'))
  assert m['png_sha256']==sha(p) and m['GT_used_in_inference'] is False and m['checkpoint_sha256']==complete['checkpoint_sha256']
  images.append(dict(id=row['id'],seed=seed,path=str(p),sha256=sha(p)))
write(R/'BF_output_hashes.json',images)
weight=Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-4df8aa71.pth'
assert sha(weight)=='4df8aa717fd11ab2d34e9483c5984ef7cdbefd732366a836076020db08909c9a'
calibration=Path(lpips.__file__).parent/'weights/v0.1/alex.pth'
assert sha(calibration)=='df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0'
torch.set_num_threads(2)
metric=lpips.LPIPS(net='alex',pnet_rand=True).eval().requires_grad_(False)
pretrained=torch.load(weight,map_location='cpu')
metric.net.load_state_dict({k:pretrained['features.'+k.split('.',1)[1]] for k in metric.net.state_dict()},strict=True)
metric=metric.cuda()
tensor=lambda im:torch.from_numpy(np.asarray(im,np.float32).transpose(2,0,1)/127.5-1)[None].cuda()
write(out/'contract.json',dict(torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),alexnet_sha256=sha(weight),calibration_sha256=sha(calibration),source_sha256=sha(__file__),LPIPS_atol=1e-6,regional_atol=1e-12,GT_LPIPS_resize='BILINEAR384x512',GT_legacy_Leak_resize='BICUBIC384x512',legacy_preserved=True))
def score(path,row):
 im=Image.open(path).convert('RGB');assert im.size==SIZE
 gt0=Image.open(row['gt']).convert('RGB');gt=gt0.resize(SIZE,Image.Resampling.BILINEAR)
 t=mask_image_to_bool(build_sketch_garment_mask(Image.open(row['sketch']),*SIZE)[0]);assert t.any()
 fg=mask_image_to_bool(estimate_cloth_foreground_mask(im,*SIZE)[0]);outside=~(cv2.dilate(t.astype('uint8'),np.ones((9,9),np.uint8))>0)
 sat=np.asarray(im.convert('HSV'),np.float32)[:,:,1]/255
 delta=np.linalg.norm(np.asarray(im,np.float32)-np.asarray(gt0.resize(SIZE,Image.Resampling.BICUBIC),np.float32),axis=2)/math.sqrt(3*255*255)
 return dict(lpips_gt=float(metric(tensor(im),tensor(gt)).mean()),contour_f1=ns['contour_f1'](fg,t),iou=float((fg&t).sum()/max(1,(fg|t).sum())),leak=float(((sat>.16)|(delta>.08))[outside].mean()))
with torch.inference_mode():
 check=[];max_errors={k:0. for k in ['lpips_gt','contour_f1','iou','leak']}
 for row in rows:
  for arm in ['E5','B1','M10','M10_GAIN','M12','M12_GAIN']:
   for seed in SEEDS:
    p=OLD/'outputs'/arm/row['id']/('seed%d.png'%seed);v=score(p,row);expected=index[arm,row['id'],seed]
    errors={k:abs(v[k]-expected[k]) for k in max_errors}
    max_errors={k:max(max_errors[k],errors[k]) for k in max_errors}
    check.append(dict(id=row['id'],variant=arm,seed=seed,png_sha256=sha(p),**v,errors=errors))
  print('M12_GPU_REPLAY',row['id'],len(check),flush=True)
 write(out/'M12_GPU_replay.json',check)
 passed=all(v<=(1e-6 if k=='lpips_gt' else 1e-12) for k,v in max_errors.items())
 write(out/'M12_GPU_replay_verification.json',dict(images=len(check),passed=passed,max_errors=max_errors,formal_scores_replaced=False))
 assert passed,'先恢复旧评价实现；不得继续BF配对排名或放宽容差'
 records=[]
 for row in rows:
  for seed in SEEDS:
   p=R/'outputs/BF-Fashion'/row['id']/('seed%d.png'%seed)
   records.append(dict(id=row['id'],variant='BF-Fashion',seed=seed,png_sha256=sha(p),**score(p,row)))
  print('BF_GPU_EVAL',row['id'],len(records),flush=True)
write(out/'per_image.json',records)
write(out/'complete.json',dict(images=len(records),M12_GPU_replay_passed=True,new_training=0,source_sha256=sha(__file__)))
print('BF_GPU_EVALUATION_COMPLETE',len(records),flush=True)
