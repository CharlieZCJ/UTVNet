# UTVNet: Adaptive Unfolding Total Variation Network for Low-Light Image Enhancement  
![image](https://github.com/CharlieZCJ/UTVNet/blob/main/UTVNet%20.png)  

Real-world low-light images suffer from two main degradations, namely, inevitable noise and poor visibility. Since the noise exhibits different levels, its estimation has been implemented in recent works when enhancing low-light images from raw Bayer space. When it comes to sRGB color space, the noise estimation becomes more complicated due to the effect of the image processing pipeline. Nevertheless, most existing enhancing algorithms in sRGB space only focus on the low visibility problem or suppress the noise under a hypothetical noise level, leading them impractical due to the lack of robustness. To address this issue, we propose an adaptive unfolding total variation network (UTVNet), which approximates the noise level from the real sRGB low-light image by learning the balancing parameter in the model-based denoising method with total variation regularization. Meanwhile, we learn the noise level map by unrolling the corresponding minimization process for providing the inferences of smoothness and fidelity constraints. Guided by the noise level map, our UTVNet can recover finer details and is more capable to suppress noise in real captured low-light scenes. Extensive experiments on real-world low-light images clearly demonstrate the superior performance of UTVNet over state-of-the-art methods. This repo is the official implementation of ICCV2021 paper "Adaptive Unfolding Total Variation Network for Low-Light Image Enhancement". Our Codes and models are coming soon. For more details, please see our [paper](https://arxiv.org/abs/2110.00984).

## Code
### Environment
Training is implemented with PyTorch. This code was developed under Python 3.6 and Pytorch 1.1.


### Data
- ELD: Download the ELD dataset from the [website](https://github.com/Vandermode/ELD), a selected sRGB version used in our model [website]()  

- sRGBSID: Contact the author by [website](https://openaccess.thecvf.com/content_CVPR_2020/html/Xu_Learning_to_Restore_Low-Light_Images_via_Decomposition-and-Enhancement_CVPR_2020_paper.html).

### Evaluation

## Citation
If you find our work useful in your research, please consider citing:
```
@InProceedings{Zheng_2021_ICCV,
    author    = {Zheng, Chuanjun and Shi, Daming and Shi, Wentian},
    title     = {Adaptive Unfolding Total Variation Network for Low-Light Image Enhancement},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {4439-4448}
}
```

