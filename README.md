# Waseda-ColorizationNet
## Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification
### Satoshi Iizuka, Edgar Simo-Serra, and Hiroshi Ishikawa (ACM Transaction on Graphics Proc. of SIGGRAPH 2016)

### About this repositry 
- This repository is a PyTorch reimplementation of the [Let there be Color!](https://dl.acm.org/doi/pdf/10.1145/2897824.2925974) model.
  
### Installation
```
git clone https://github.com/Shunta-Shimizu/Waseda-ColorizationNet.git
cd Waseda-ColorizationNet
conda create -n waseda_color python=3.8.0
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install opencv-python numpy scikit-image
```

### Dataset
- Places365-Standard
- ImageNet 

### Train
```
python train.py                \
 --train_data_dir /xxx/yyy/zzz/ \
 --val_data_dir /aaa/bbb/       \
 --save_model_dir ./model/      \
 --batch_size n                 \
 --num_epochs m
````

### Test
```
python test.py --test_data_dir /xxx/yyy/wwww/ --model_path ./model/xxx.pth --save_result_dir /aaa/bbb/
```

### Tasks
