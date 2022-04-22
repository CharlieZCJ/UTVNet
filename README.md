# UTVNet: Adaptive Unfolding Total Variation Network for Low-Light Image Enhancement  


This repo is the official implementation of ICCV2021 paper "Adaptive Unfolding Total Variation Network for Low-Light Image Enhancement". For more details, please see our [paper](https://arxiv.org/abs/2110.00984). 


## UTVNet

![image](https://github.com/CharlieZCJ/UTVNet/blob/main/UTVNet%20.png)  


 Real-world low-light images suffer from two main degradations, namely, inevitable noise and poor visibility. Since the noise exhibits different levels, its estimation has been implemented in recent works when enhancing low-light images from raw Bayer space. When it comes to sRGB color space, the noise estimation becomes more complicated due to the effect of the image processing pipeline. Nevertheless, most existing enhancing algorithms in sRGB space only focus on the low visibility problem or suppress the noise under a hypothetical noise level, leading them impractical due to the lack of robustness. To address this issue, we propose an adaptive unfolding total variation network (UTVNet), which approximates the noise level from the real sRGB low-light image by learning the balancing parameter in the model-based denoising method with total variation regularization. Meanwhile, we learn the noise level map by unrolling the corresponding minimization process for providing the inferences of smoothness and fidelity constraints. Guided by the noise level map, our UTVNet can recover finer details and is more capable to suppress noise in real captured low-light scenes. Extensive experiments on real-world low-light images clearly demonstrate the superior performance of UTVNet over state-of-the-art methods.

## Code
### Prerequisites
- Python 3.6 and Pytorch 1.7.1.
- Requirements: Numpy, skiamge, tqdm, opencv-python, matplotlib, pillow

### Data
ELD dataset
- ELD dataset [website](https://github.com/Vandermode/ELD),
-  Download the selected sRGB version used in our model [website](https://drive.google.com/drive/folders/141j0O4d2k35aOADWHHfJY7idiHNI21Nm?usp=sharing)  
- Copy the dataset to ```dataset/ELD```

sRGBSID
- Contact the author by [website](https://openaccess.thecvf.com/content_CVPR_2020/html/Xu_Learning_to_Restore_Low-Light_Images_via_Decomposition-and-Enhancement_CVPR_2020_paper.html).
- Copy the Test and Ground-Truth dataset to ```dataset/sRGBSID```
### Model
Download the pretrained model from  [website](https://drive.google.com/drive/folders/1uWfnaNd9Yy6hhDH8uNvfeC1Sxfz4QCZL?usp=sharing), copy the model to ```pretrain_model```
### Evaluation

sRGBSID  

```
python test.py --data_name='sRGBSID'
python evaluate.py --data_name='sRGBSID'
```
please check the results in ```result/sRGBSID```  


ELD

- Sony 
```
python test.py --data_name='ELD_sony'
python evaluate.py --data_name='ELD_sony'
```
please check the results in ```result/ELD_sony```    


- Cano 
```
python test.py --data_name='ELD_cano'
python evaluate.py --data_name='ELD_cano'
```
please check the results in ```result/ELD_cano```      

- Niko
```
python test.py --data_name='ELD_niko'
python evaluate.py --data_name='ELD_niko'
```
please check the results in ```result/ELD_niko```      

### Test Result
If you find it difficult to generate the test results, plesase send me an email.(<chuanjunzhengcs@gmail.com>)


## Citation
If you find our work useful in your research, please consider citing:
```bibtex
@InProceedings{Zheng_2021_ICCV,
    author    = {Zheng, Chuanjun and Shi, Daming and Shi, Wentian},
    title     = {Adaptive Unfolding Total Variation Network for Low-Light Image Enhancement},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {4439-4448}
}
```

## Acknowledgements
The code in ```basicblock.py``` is based on Model [USRNet](https://github.com/cszn/USRNet)
