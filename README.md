<div align="center">

# AerialMegaDepth: Learning Aerial-Ground Reconstruction and View Synthesis

[Khiem Vuong](https://www.khiemvuong.com/), [Anurag Ghosh](https://anuragxel.github.io/), [Deva Ramanan*](https://www.cs.cmu.edu/~deva), [Srinivasa Narasimhan*](https://www.cs.cmu.edu/~srinivas), [Shubham Tulsiani*](https://shubhtuls.github.io/)\
${{\color{Red}\Huge{\textsf{  CVPR\ 2025\ \}}}}\$

[[`arXiv`](https://arxiv.org/abs/XXXX.XXXXX)]
[[`Project Page`](https://aerial-megadepth.github.io/)]
[[`Bibtex`](#reference)]
</div>


## Get Started
We provide DUSt3R checkpoint fine-tuned on our AerialMegaDepth dataset. It is fully compatible with the original DUSt3R codebase - if you already have DUSt3R set up, you can simply swap the checkpoint! Below, we provide the full set-up instructions (mostly copied from [DUSt3R repo](https://github.com/naver/dust3r)).

### Installation

1. Clone the repository:
```bash
git clone --recursive https://github.com/kvuong2711/aerial-megadepth.git
cd aerial-megadepth

# If you already cloned the repository, you can update the submodules:
# git submodule update --init --recursive
```
2. Create environment and install dependencies:
```bash
conda create -n aerialmd python=3.11 cmake=3.14.0
conda activate aerialmd 
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
pip install -r dust3r/requirements.txt
pip install -r dust3r/requirements_optional.txt
```
3. Optional, compile the cuda kernels for RoPE (as in CroCo v2):
```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../
```
### Checkpoints

To download the DUSt3R checkpoint finetuned on our AerialMegaDepth dataset:

```bash
cd dust3r
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/
```

### Inference

To run the inference code, you can use the following command:

```bash
python demo.py --
```


## Data Generation
For data generation, please refer to the [data_generation](data_generation) folder.


## Issues
If you have trouble preparing the dataset, feel free to reach out to me via [email](mailto:kvuong@andrew.cmu.edu).

## Reference

If you find our work to be useful in your research, please consider citing our paper:

```bibtex
@inproceedings{vuong2025aerialmegadepth,
  title={AerialMegaDepth: Learning Aerial-Ground Reconstruction and View Synthesis},
  author={Vuong, Khiem and Ghosh, Anurag and Ramanan, Deva and Narasimhan, Srinivasa and Tulsiani, Shubham},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025},
}
```