# Caricature-Photo-Recognition
Match Caricature and photo pairs. It is an interesting task, isn't it?

## The experiments are conducted on Pytorch 0.4.0.

* dataset.py --Preprocess the images. 

* net_sphere.py --Added my CBMA module.

* transfer learning.py --Use transfer learning to get the traning model of SphereNet. 

* utils.py --Some other functions.

* wc_eval.py --Test the model and evaluate our results. 


The dataset is very large. If you want to try it, you can download it from [WebCaricature](https://cs.nju.edu.cn/rl/WebCaricature.htm)


I prepocessed that by using a face recognition tool to get the face area in a caricature . If it can't do that, it would return NULL so I considered those too abstract caricatures as noise and added them into training set to make it more robust. 


Based on [SephereNet](http://www.cvlibs.net/publications/Coors2018ECCV.pdf), I added CBAM (Convolutional Block Attention Module) to extract features on each pair of image.


It achieved high accuracy and outperformed the other algorithms. Our paper *"Gated Fusion of Discriminant Features for Caricature Recognition"* has been accepeted by IScIDE2019, which you can read later. 





