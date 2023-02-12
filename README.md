# AUDIO GENDER RECOGNITION 


## NON-TECHNICAL DESCRIPTION

This project aims to create a software program, based on a Convolutional Neural Network, which can 
distinguish male from female voices based on a short audio recording of a human voice. Upon receiving
an audio input, the software will return a classification 'male' or 'female'. 

The software has been trained using a large number of recordings of human adults saying a single digit i.e. 0-9. 

## DATA

The data used to train this solution is  AudioMNIST. It is widely used to train deep and/or convolutional neural 
networks for speech-recognition tasks. In this project the dataset is used to train a CNN to classify audio input as 
either male or female depending on the speaker.

The dataset was created by Becker et al and is available here: https://github.com/soerenab/AudioMNIST.
The creators of the dataset request that projects using their work cite the following paper:

|          |                        |
| ------------- |------------------------------------------------------------------------------------------------------------|
| Title         | Interpreting and Explaining Deep Neural Networks for Classification of Audio Signals                       |
| Journal       | CoRR |
| Volume        | abs/1807.03418 |
| Year          | 2018 |
| ArchivePrefix | arXiv |
| Eprint        | 1807.03418 |
| Authors       | Becker, Sören and Ackermann, Marcel and Lapuschkin, Sebastian and Müller, Klaus-Robert and Samek, Wojciech |

These data must be preprocessed in order to make them suitable for processing by a convolutional neural network.
Preprocessing includes adding a specific form of emphasis, as well as padding. Padding ensures that the resulting input 
has the same length irrespective of audio sample. 
The preprocessing code in included in this project in the Jupyter notebook 
```preprocess_wav.ipynb```.  

## MODEL 

The model chosen to perform this task is a Convolutional Neural Network (CNN). The CNN does not process audio samples
(in the ```wav``` format) directly, but instead processes an image representation (in the ```jpg``` format) of the 
original audio. The image represents a so-called Mel-frequency cepstrum (MFC). 
For example, the following MFC image represents a female saying the digit "zero": 

![img_1.png](img_1.png)
   
These greyscale images have shape 98x12x1, and the CNN is specifically designed to process shapes of this kind.                                

## HYPERPARAMETER OPTIMSATION

TODO 

Description of which hyperparameters you have and how you chose to optimise them. 

## RESULTS

TODO

A summary of your results and what you can learn from your model 


## CONTACT DETAILS

Mike Mannion
michaelmannion@me.com

 

