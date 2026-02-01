# BF-Fashion-
BF-Fashion复现

1. 环境
git clone https://github.com/zibingo/BF-Fashion.git
cd BF-Fashion
conda env create -f environment.yaml
conda activate ldm
pip install -e .
cd src/taming-transformers
pip install -e .
cd ../..
下载预训练自编码模型，解压到“models\first_stage_models\vq-f8”

2. 数据准备
下载已清理的类型感知数据集，并应用HED生成草图。https://drive.google.com/file/d/1TCGg_vj79rmJrUW8J19HcyAkGcIXrNUJ/view

您也可以将我们处理后的数据集下载到项目文件夹的根目录。https://drive.google.com/file/d/1E5HW_17IfjfkBlEPLCaE6mbjPtWvJU2H/view

3. 修改配置文件
修改配置文件中的绝对路径 

sed -i "s|/workspace/BF-Fashion|$(pwd)|g" configs/latent-diffusion/mydata_ldm-vq-f8.yaml
sed -i "s|/workspace/BF-Fashion|$(pwd)|g" configs/latent-diffusion/mydata_ldm-vq-f8_test.yaml
4. 培训
sh run_train.sh
5. 推断
在这里下载检查点，并放在项目文件夹的根目录中 https://drive.google.com/file/d/1TCGg_vj79rmJrUW8J19HcyAkGcIXrNUJ/view

sh run_batch_sample.sh
6. 演示
python gradio_img2img.py
7. 计算指标
python m_class_fid_lpips.py
python m_class_histo_loss.py


