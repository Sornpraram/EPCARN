# Real-time Lightweight CARN Super-Resolution with Edge Prior

### About this work
This is an implementation of Super-Resolution model call "CARN" and Edge prior. We design this model for a real world task
by integrate into Real-time Streaming from one place to other using WIFI network.
(This work is still on progress)

### Requirements
- Python 3
- PyTorch
- Numpy, Scipy
- Pillow, Scikit-image

### Dataset
We retrive dataset of 2k 10000 from real world along with DIV2K dataset.

### Test Pretrained Models
We provide the pretrained models in `checkpoint` directory. To test CARN on benchmark dataset:
```shell
$ python carn/sample.py --model carn \
                        --test_data_dir dataset/<dataset> \
                        --scale [2|3|4] \
                        --ckpt_path ./checkpoint/<path>.pth \
                        --sample_dir <sample_dir>
```
and for CARN-M,
```shell
$ python carn/sample.py --model carn_m \
                        --test_data_dir dataset/<dataset> \
                        --scale [2|3|4] \
                        --ckpt_path ./checkpoint/<path>.pth \
                        --sample_dir <sample_dir> \
                        --group 4
```
We provide our results on four benchmark dataset (Set5, Set14, B100 and Urban100). [Google Drive](https://drive.google.com/drive/folders/1R4vZMs3Adf8UlYbIzStY98qlsl5y1wxH?usp=sharing)

### Training Models
Here are our settings to train CARN and CARN-M. Note: We use two GPU to utilize large batch size, but if OOM error arise, please reduce batch size.
```shell
# For CARN
$ python carn/train.py --patch_size 64 \
                       --batch_size 64 \
                       --max_steps 600000 \
                       --decay 400000 \
                       --model carn \
                       --ckpt_name carn \
                       --ckpt_dir checkpoint/carn \
                       --scale 0 \
                       --num_gpu 2
# For CARN-M
$ python carn/train.py --patch_size 64 \
                       --batch_size 64 \
                       --max_steps 600000 \
                       --decay 400000 \
                       --model carn_m \
                       --ckpt_name carn_m \
                       --ckpt_dir checkpoint/carn_m \
                       --scale 0 \
                       --group 4 \
                       --num_gpu 2
```
In the `--scale` argument, [2, 3, 4] is for single-scale training and 0 for multi-scale learning. `--group` represents group size of group convolution. The differences from previous version are: 1) we increase batch size and patch size to 64 and 64. 2) Instead of using `reduce_upsample` argument which replace 3x3 conv of the upsample block to 1x1, we use group convolution as same way to the efficient residual block.

### Results
**Note:** As pointed out in [#2](https://github.com/nmhkahn/CARN-pytorch/issues/2), previous Urban100 benchmark dataset was incorrect. The issue is related to the mismatch of the HR image resolution from the original dataset in x2 and x3 scale. We correct this problem, and provided dataset and results are fixed ones.

