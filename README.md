# Gaussian Splatting with NeRF-based Color and Opacity
Dawid Malarz*, Weronika Smolak*, Jacek Tabor, Sławomir Tadeja, Przemysław Spurek (* indicates equal contribution)<br>
<img src=assets/Directed_nerf_all.png height="300" >  <img src=assets/Directed_nerf.png height="300">

| arXiv |
| :---- |
| [Gaussian Splatting with NeRF-based Color and Opacity](https://arxiv.org/pdf/2312.13729.pdf)|


Abstract: *Neural Radiance Fields (NeRFs) have demonstrated the remarkable potential of neural networks to capture the intricacies of 3D objects. By encoding the shape and color information within neural network weights, NeRFs excel at producing strikingly sharp novel views of 3D objects. Recently, numerous generalizations of NeRFs utilizing generative models have emerged, expanding its versatility. In contrast, Gaussian Splatting (GS) offers a similar renders quality with faster training and inference as it does not need neural networks to work. We encode information about the 3D objects in the set of Gaussian distributions that can be rendered in 3D similarly to classical meshes. Unfortunately, GS are difficult to condition since they usually require circa hundred thousand Gaussian components. To mitigate the caveats of both models, we propose a hybrid model that uses GS representation of the 3D object's shape and NeRF-based encoding of color and opacity. Our model uses Gaussian distributions with trainable positions (i.e. means of Gaussian), shape (i.e. covariance of Gaussian), color and opacity, and neural network, which takes parameters of Gaussian and viewing direction to produce changes in color and opacity. Consequently, our model better describes shadows, light reflections, and transparency of 3D objects.*
<img src=assets/garden.png width="900">
## 3D Gaussian Splatting
Code is a modification of the original code from [3D Gaussian Splatting for Real-Time Radiance Field Rendering paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

## Requirements and how to use

Dependencies for Conda environment management are stored in `environment.yml`.

In order to train the main model (VDGS opacity multiplication) just simply run the command below, similarly like you would in [the original repository for 3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/):
```
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --vdgs_type opacity --vdgs_operator mul 
```
In order to use other models mentioned in VDGS paper from our ablation study such as these:
- Default model: VDGS opacity multiplication 
- VDGS color (SHs) factor sum
- VDGS color (SHs) factor multiplication
- VDGS opacity factor multiplication
- VDGS opacity and color (SHs) factors sum
- VDGS opacity and color (SHs) factors multiplication 

You can use flags such as `--vdgs_operator` and `--vdgs_type` in order to specify which model you want to use. 

Supported arguments: 
- `--vdgs_type`: "both", "opacity", "color" 
- `--vdgs_operator`: "mul", "add" 

To evaluate the model use commands as described in [the original repository 3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) with additional flags for the model that you wish to use.
```
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval --vdgs_type opacity --vdgs_operator mul # Train with train/test split
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```
Other optional arguments work the same as in [the original repository 3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

## Notes

Interactive viewers such as SIBR viewers are not currently supported by our code and may be rendered incorrectly by them.

