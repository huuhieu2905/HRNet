# High-resolution networks (HRNets) for facial landmark detection

## Quick start
#### Environment
This code is developed using on Python 3.8 and PyTorch 2.3.0 on Ubuntu 20.04 with NVIDIA GPUs. 

#### Install
1. Install dependencies
````bash

pip install -r requirements.txt
````
2. Clone the project
````bash 
git clone https://github.com/huuhieu2905/HRNet.git
````

#### HRNetV2 pretrained models
Download pretrained models: [OneDrive](https://onedrive.live.com/?authkey=%21AMkPimlmClRvmpw&id=F7FD0B7F26543CEB%21112&cid=F7FD0B7F26543CEB&parId=root&parQt=sharedby&o=OneUp)
Download model trained on 300W dataset: [OneDrive](https://onedrive.live.com/?authkey=%21AILmza1XU%2D4WhnQ&id=735C9ADA5267A325%21121&cid=735C9ADA5267A325&parId=root&parQt=sharedby&o=OneUp)
```bash
cd HRNet-Facial-Landmark-Detection
mkdir hrnetv2_pretrained
# Download pretrained models into this folder
```
#### Data

1. You need to download data from [OneDrive](https://husteduvn-my.sharepoint.com/:u:/g/personal/hieu_nh215577_sis_hust_edu_vn/EZIcld4MRKROqEg8-Jg029kBWmoDRCPTJSZLXIDjQT3NqQ?e=BCF5er)
Your `data` directory should look like this:

````
HRNet-Facial-Landmark-Detection
-- lib
-- tools
-- data
   |-- 300w
   |   |-- face_landmarks_300w_test.csv
   |   |-- face_landmarks_300w_train.csv
   |   |-- face_landmarks_300w_valid.csv
   |   |-- images

````

#### Train
````bash
python train.py --cfg <CONFIG-FILE>
# example:
python train.py --cfg config.yaml
````

#### Test
````bash
python test.py --cfg <CONFIG-FILE> --model-file <MODEL WEIGHT> 
# example:
python test.py --cfg config.yaml --model-file HR18-WFLW.pth
````
#### Inference
````bash
python inference.py
````

## Reference
[1] Deep High-Resolution Representation Learning for Visual Recognition. Jingdong Wang, Ke Sun, Tianheng Cheng, 
    Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, Wenyu Liu, Bin Xiao. Accepted by TPAMI.  [download](https://arxiv.org/pdf/1908.07919.pdf)
[2]
