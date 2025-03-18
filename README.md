# Semantic-Segmentation

In this section I shortly intriduce the semantic segmentation neural network models. Right now there are only 3 models: U-Net, LinkNet, PSPNet. 

Semantic segmentation is a computer vision method that not only detects objects in images but also determines their exact spatial location by classifying each pixel. This technique is useful when an image needs to be divided into multiple categories (multiclass classification) rather than just distinguishing between two classes (binary classification). Convolutional neural networks (CNNs) were specifically developed for image processing and have shown higher accuracy than traditional multilayer perceptrons (MLPs), while also requiring fewer trainable parameters. A major breakthrough in CNN-based image classification happened with the introduction of AlexNet, which won the ImageNet competition in 2012. Later, more advanced models were developed, including VGG, ResNet, and GoogleNet. However, these models were mostly designed for whole-image classification rather than pixel-level seg-
mentation. For semantic segmentation, the U-Net architecture was introduced. Originally designed for medical image analysis, U-Net has an encoder-decoder structure, where the encoder extracts image features, and the decoder reconstructs spatial information to produce a segmentation map. Other architectures have also been proposed, such as LinkNet, which is optimized for real-time applications, and PSPNet, which uses a pyramidal pooling structure (PPM) to capture both local and global image features.


![image](https://github.com/user-attachments/assets/ed5950a2-1927-4612-b98b-14045c5dfd50)
