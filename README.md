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


**Use a CNN to Classify Dog Breeds (using Transfer Learning)**

I used a CNN to Classify Dog Breeds from pre-trained VGG-16 model **with test accuracy: 38.1579 %.**

The model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model. We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.

**Create a CNN to Classify Dog Breeds (using Transfer Learning)**

I then used Transfer learning to create a CNN that can identify dog breed from images **with 80.1435 % accuracy on the test set.**

My final CNN architecture is built with the Resnet50 bottleneck. Further, GlobalAveragePooling2D used to flatten the features into vector. These vectors were fed into the fully-connected layer towards the end of the ResNet50 model. The fully-connected layer contains one node for each dog category and is assisted with a softmax function.

# Result Section :

The use of 'transfer learning - Resnet50 model' to implement an algorithm for a Dog identification application has been demonstrated here. The user can provide an image, and the algorithm first detects whether the image is human or dog. If it is a dog, it predicts the breed. If it is a human, it returns the resembling dog breed. The model produces the test accuracy of around 80%. The scope of further improvements has also been suggested in this work.

Here are examples of the algorithms:

![Screenshot](result2.png)



![Screenshot](result3.png)



![Screenshot](result1.png)



# Hey guys.!! The step-wise thought process of creating this model and its results are available here in my [blog](https://medium.com/@manishislampur1988/fun-with-cnn-app-to-identify-breed-of-your-doggy-9d3dbd06c513)
