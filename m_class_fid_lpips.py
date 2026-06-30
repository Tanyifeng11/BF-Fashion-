from pytorch_fid import fid_score
import argparse
import lpips
import os
from PIL import Image
from tqdm import tqdm 
import threading
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

PROJECT_DIR = "/share/home/u2515283058/BF-Fashion-"
BASE_PATH = "/share/home/u2515283058/datasets/BF/test"
EVAL_OUTPUT_DIR = os.path.join(PROJECT_DIR, "eval_output")
GENERATED_BASE_PATH = os.path.join(EVAL_OUTPUT_DIR, "test")
OUTPUT_FILE = os.path.join(EVAL_OUTPUT_DIR, "m_class_fid_lpips.txt")
DEFAULT_CLASSES = ["bag", "top", "outwear", "pants", "dress"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--output-file", default=OUTPUT_FILE)
    return parser.parse_args()


def get_target_folder(class_folder):
    gt_folder = os.path.join(class_folder, "gt")
    if os.path.isdir(gt_folder):
        return gt_folder
    return os.path.join(class_folder, "cloth")


def compute_fid(real_images_folder,generated_images_folder,results):
    print("computing fid...")
    fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],
                                                    device="cuda",
                                                    batch_size=1,
                                                    dims=2048)
    fid_value = round(fid_value,3)
    print("fid:",fid_value)
    results["fid"] = fid_value

def compute_lpips(real_images_folder,generated_images_folder,results):
    print("computing lpips...")
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    real_images_name = os.listdir(real_images_folder)
    lpips_alex_total = 0.0

    for filename in tqdm(real_images_name,desc="LPIPS"):
        img1 = Image.open(os.path.join(real_images_folder, filename))
        img2 = Image.open(os.path.join(generated_images_folder,filename))
        img1 = (2.0 * transform(img1) - 1.0).unsqueeze(0).cuda()
        img2 = (2.0 * transform(img2) - 1.0).unsqueeze(0).cuda()
        d1 = loss_fn_alex.forward(img1, img2)
        lpips_alex_total += d1.item()

    lpips_alex_avg = round(lpips_alex_total/len(real_images_name),3)
    print("lpips_alex_avg:",lpips_alex_avg)
    results["lpips_alex_avg"] = lpips_alex_avg

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    classnames = args.classes
    if args.output_file == OUTPUT_FILE and len(classnames) == 1:
        args.output_file = os.path.join(EVAL_OUTPUT_DIR, "metrics", f"fid_lpips_{classnames[0]}.txt")
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(args.output_file):
        os.remove(args.output_file)
    # classnames = ["dress"]
    for classname in classnames:
        class_folder = os.path.join(BASE_PATH, classname)
        real_images_folder = get_target_folder(class_folder)
        generated_images_folder = os.path.join(GENERATED_BASE_PATH, classname, "gen")
        results = {}
        thread1 = threading.Thread(target=compute_fid, args=(real_images_folder,generated_images_folder,results))
        thread2 = threading.Thread(target=compute_lpips, args=(real_images_folder,generated_images_folder,results))
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        data = [results["fid"],results["lpips_alex_avg"]]

        # 打开文件并追加写入数据
        with open(args.output_file, 'a') as file:
            file.write(','.join(map(str, [classname] + data)) + '\n')
