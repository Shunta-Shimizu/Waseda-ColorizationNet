# Waseda-ColorizationNet
## Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification (SIGGRAPH 2016)
### Satoshi Iizuka, Edgar Simo-Serra, and Hiroshi Ishikawa (ACM Transaction on Graphics Proc. of SIGGRAPH 2016)

### About this repositry 
- This repository is a PyTorch reimplementation of the [Let there be Color!](https://dl.acm.org/doi/pdf/10.1145/2897824.2925974) model.
  
### Installation
```
git clone https://github.com/Shunta-Shimizu/DDPM.git
cd DDPM
conda create -n ddpm python=3.9
pip install requirements.txt
```

### Dataset
- Places365-Standard
- ImageNet 

### Train
```
python train.py --train_data_dir ./ --save_model_dir ./ 
````

### Test
```
python test.py --test_data_dir ./ --model_path ./ --save_result_dir ./
```

### Tasks
