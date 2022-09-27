# Deep Rectangling

## Students Team Data

Kareem Jabareen - [LinkedIn](https://www.linkedin.com/in/kareem-mokhtar-jabareen/)

Malik Egbarya - [LinkedIn](https://www.linkedin.com/in/malikegbarya/)

## Project Description

In this work we are trying to improve the results of a [recent work](https://github.com/nie-lang/DeepRectangling).

We are creating augmentations in different ways for each image in the input dataset, this way we are training our model on a much bigger dataset with a good chance for better accuracy.

## DIR-D

For getting the dataset used to train and test our work, you can download it from [Google Drive](https://drive.google.com/file/d/1Q0b1OLa3qbHGhFVwsLJ_fkgTRlAzFPCB/view?usp=sharing).

## Requirements

During our work, we had our code on:

- Windows 10 64-bit
- Nvidia GeForce GTX 1650
- CUDA 10.1
- Python 3.6 interpreter in Anaconda environment

In addition, we had many essential Python packages and dependencies for running the code that you can find in the [env file](./env.yaml).

## Training

### Step 1: Download the pretrained vgg19 model

Download [VGG-19](https://www.vlfeat.org/matconvnet/pretrained/#downloading-the-pre-trained-models). Search imagenet-vgg-verydeep-19 in this page and download imagenet-vgg-verydeep-19.mat.

### Step 2: Train the network

Modify the 'Codes/constant.py' to set the 'TRAIN_FOLDER'/'ITERATIONS'/'GPU'. In our experiment, we set 'ITERATIONS' to 100,000.

```
cd Codes/
python train.py
```

## Testing

### Pretrained model for deep rectangling

Our pretrained rectangling model can be available at [Google Drive](https://drive.google.com/file/d/1MYtFgnnVC0iqnEyeN74tgezsPsXAgkPH/view?usp=sharing). Extract the compressed file and place the four files to 'Codes/checkpoints/Ptrained_model/' folder.

### Testing

Modify the 'Codes/constant.py' to set the 'TEST_FOLDER'/'GPU'. The path for the checkpoint file can be modified in 'Codes/inference.py'.

```
cd Codes/
python inference.py
```

### Testing with arbitrary resolution images

Modify the 'Codes_for_Arbitrary_Resolution/constant.py'to set the 'TEST_FOLDER'/'GPU'. The path for the checkpoint file can be modified in 'Codes_for_Arbitrary_Resolution/inference.py'. Then, put the testing images into the folder 'Codes_for_Arbitrary_Resolution/other_dataset/' (including input and mask) and run:

```
cd Codes_for_Arbitrary_Resolution/
python inference.py
```

The rectangling results can be found in Codes_for_Arbitrary_Resolution/rectangling/.

## Reference

1. Lang Nie, Chunyu Lin, Kang Liao, Shuaicheng Liu, and Yao Zhao. Depth-aware multi-grid deep homography estimation with contextual correlation. IEEE Trans. on Circuits and Systems for Video Technology, 2021.

2. Kaiming He, Huiwen Chang, Jian Sun. Rectangling panoramic images via warping. SIGGRAPH, 2013.