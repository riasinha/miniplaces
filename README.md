# MiniPlaces Challenge 
## Project Overview
The [MiniPlaces Challenge](https://github.com/CSAILVision/miniplaces) was introduced in a Computer Vision class at MIT. The goal of the project is to classify images into one of 100 scene categories (e.g. hotel, bridge, restaurant, etc).
## Dataset
The link for downloading the image data is [here](http://miniplaces.csail.mit.edu/data/data.tar.gz). The image database statistics is as follows:
```
  Training: 	100,000 images, with 1000 images per category
  Validation:	10,000 images, with 100 images per category
  Test:		10,000 images, with 100 images per category
```
The images have been resized to 128x128.

## Model Selection
For this project, I researched different [Keras Image Classification](https://keras.io/api/applications/) neural networks to use as a baseline model. I selected the [Xception](https://keras.io/api/applications/xception/) model. It was 88 MB, included 22,910,480 parameters, and had a Top-5 accuracy rate of 95% on the ImageNet dataset. The model is based on the [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) paper. I selected this model because it had a good balance between complexity and accuracy. 

## Transfer Learning
My main idea was to start with a base Xception model in Keras that was pre-trained on the [ImageNet](http://www.image-net.org/) dataset. The base Xception model uses 299x299 sized images for 1000 classes. I customized the input layer for 128x128 sized images. I added two Dense layers and used the softmax activation function as output of the model to classify into one of the 100 classes. The initialized the base Xception model with pre-trained weights and my custom top layer with random weights. 

Transfer learning was done in 2 stages. In the first stage, the base Xception model parameters were frozen and only the top layer was trained using a bigger learning rate. Subsequently, all layers were trained using a finer learning rate. Some of these ideas are discussed [here](https://keras.io/guides/transfer_learning/)

## Model Training 
The training images were imported as batches of tensors using [ImageDataGenerator](https://keras.io/api/preprocessing/image/). I used a batch size of 128 images. Each input batch was of shape (128, 128, 128, 3) and each output batch was of shape (128, 100). Also plotted a few images to preview data. 

For stage 1 coarse training, the model was compiled with the Adam optimizer and a coarse learing rate (1e-3) using the categorical crossentropy loss. Tthe base model layers were set to non-trainable and only the custom top layer was trained. After 5 epochs, the trained model converged to an Top-1 accuracy of 64% and a Top-5 accuracy of 88%.

For stage 2 fine training, the model was compiled with the Adam optimizer and a finer learing rate (1e-4) using the categorical crossentropy loss. All model parameters were set to be trainable. After 5 epochs, the fully trained model converged to an Top-1 and a Top-5 accuracy close to 100%.

## Validation and Testing

## Results and Conclusions
