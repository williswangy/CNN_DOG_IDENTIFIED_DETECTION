# Use of CNN and Transfer Learning for a Dog Identification App

# Project Overview

At the end of this project, the app will  accept any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling, what a joke?

# Project Motivation

Though I have completed several guided projects, here I follow the requirements of the CNN project and leverage what I learned into this project.
# List of Task

The complete work has been divided into separate steps as below:

* [Step 0](#step0): Import Datasets
* [Step 1](#step1): Detect Humans
* [Step 2](#step2): Detect Dogs
* [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
* [Step 4](#step4): Use a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 5](#step5): Create a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 6](#step6): Write your Algorithm
* [Step 7](#step7): Test Your Algorithm

# Instructions :

In order to run this code on your local computer, you may need to install and download the following;

[Dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). Unzip the folder and place it in the repo, at location path/to/dog-project/dogImages.

[Human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). Unzip the folder and place it in the repo, at location path/to/dog-project/lfw. If you are using a Windows machine, you are encouraged to use 7zip to extract the folder.

[VGG-16 bottleneck](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) features for the dog dataset. Place it in the repo, at location path/to/dog-project/bottleneck_features.

[VGG-19 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) for the dog dataset. Place it in the repo, at location path/to/dog-project/bottleneck_features.

[ResNet-50 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) for the dog dataset. Place it in the repo, at location path/to/dog-project/bottleneck_features.

[Inception bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) for the dog dataset. Place it in the repo, at location path/to/dog-project/bottleneck_features.

[Xception bottleneck](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) features for the dog dataset. Place it in the repo, at location path/to/dog-project/bottleneck_features

I recommend setting up an environment for this project to ensure you have the proper versions of the required libraries.

# Installation Required:

For Mac/OSX:

	conda env create -f requirements/aind-dog-mac.yml
	source activate aind-dog
	KERAS_BACKEND=tensorflow python -c "from keras import backend"

For Linux:

	conda env create -f requirements/aind-dog-linux.yml
	source activate aind-dog
	KERAS_BACKEND=tensorflow python -c "from keras import backend"

For Windows:

	conda env create -f requirements/aind-dog-windows.yml
	activate aind-dog
	set KERAS_BACKEND=tensorflow
	python -c "from keras import backend"
 
This project requires Python 3 and the following Python libraries installed:

      NumPy
      Pandas
      matplotlib
      scikit-learn
      keras
      OpenCV
      Matplotlib
      Scipy
      Tqdm
      Pillow
      Tensorflow
      Skimage
      IPython Kernel

I recommend to install Anaconda, a pre-packaged Python distribution that contains all of the necessary libraries and software for this project


# File Descriptions

dog_app.ipynb - The file where the codes and possible solutions can be found.

bottleneck_features -When you Pick one of the above architectures, download the corresponding bottleneck features, and store the downloaded file in the bottleneck_features/ folder in the repository.

Images Folder - We will find here the images to test our algorithm. Use at least two human and two dog images.

saved_models - Where you will find the models those i worked on


*Use a CNN to Classify Dog Breeds (using Transfer Learning)**

I used a CNN to Classify Dog Breeds from pre-trained VGG-16 model **with test accuracy around : 40 %.**

The final convolutional output of the pre-trained VGG-16 model is provided as input to our model, which uses it as a fixed feature extractor. We merely add a fully connected layer and a global average pooling layer, the latter of which has a softmax and one node for each dog type.

*Create a CNN to Classify Dog Breeds (using Transfer Learning)**

I then used Transfer learning to create a CNN that can identify dog breed from images **with around 80% accuracy on the test set.**

The Resnet50 bottleneck served as the foundation for my final CNN architecture. Additionally, GlobalAveragePooling2D was employed to vectorize the features. At the very end of the ResNet50 model, these vectors were input into the fully-connected layer. Each dog category is represented by a single node in the fully connected layer, which also benefits from a softmax function.

# Result Section :

Here is a demonstration of how to use the "transfer learning - Resnet50 model" to implement an algorithm for a Dog identification application. An image can be submitted by the user, and the algorithm will first determine whether it is a human or a dog. It foretells the breed of dog if it is a dog. The resembling dog breed is returned if it is a human. A test accuracy of about 80% is generated by the model. This work also makes suggestions about the range of potential future improvements.


# Please refer to the medium link here [blog]()
