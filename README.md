# Declaration:  
Any commit made by the account br17011 should be attributed to the user Benoniy as this is an alternate  
account that was unintentionally used when initially uploading project files.

# Readme:
#### Introduction:
This is the readme for GPIN a simple neural network that I designed to be able to recognise images,  
originally this network was designed to identify the presence of malaria in images of blood cells which  
it managed to do with roughly 95% accuracy. I have since re-purposed it with the goal of making it a  
generic program. It should theoretically be able to identify images with greater differences than its original  
dataset with 100% accuracy after training although it has not yet been tested on another dataset.
    

## Getting started:  
#### Dependencies:  
    1. Windows 7 or Above
    
    2. Python 3.6.7  
            - numpy 1.17.4   
            - scipy 1.3.3  
            - pathlib 1.0.1
            - pillow 6.2.1
          
    3. Tensorflow 2.0.0  
            - tensorflow 2.0.0  
            - tensorflow-estimator 2.0.1

#### Important locations: 
    1. manual_testing.py and manual_training.py - "Main Program/"
    2. The dataset of images that the network uses - "Main Program/dataset/"
    3. The model that is saved after training - "Main Program/checkpoints/"

#### Installation using PIP:  
First ensure that you have python version 3.6.7 installed and working correctly along with  
your preferred package manager. Then you must install the prerequisites listed bellow that  
section making sure that the correct version are being installed. You can do this easily by  
using pip install -r requirements.txt or by manually installing the packages listed above.  

#### How to set up your dataset:  
Within the "main program" folder there is a folder called dataset, first separate your data into your  
different image types. For example if I was trying to tell the difference between apples and oranges  
then i would separate into apples and orange. After this you have to separate each of these into a  
"Training" set (80% of your images), a "Testing" set (about 15% of your images) and a "Validation" set  
(about 5% of your images).

going with the example from earlier you should organise your folder structure like this:  

![File Structure](https://github.com/Benoniy/General-Purpose-Identification-Network/blob/master/Images/file%20structure.PNG)

#### How to run the Manual Training program: 
Running the training program is as simple as running the python file manual_training.py however before  
you do this you must set up your dataset.  

First you will be asked to provide the name of a save file, if you have not trained before then you can  
enter any name that you want. If you have already trained before and want to train again then you must  
provide that save file.  
  
it is important to know how training works and you should read the later section that describes how  
it functions.

#### How to run the Manual Testing program:
Again running the testing program is as simple as running the python file manual_testing.py and providing  
the .cfg file. After the configuration file has been specified you will be asked if you want to test a  
set of random images from the models own dataset or a specific image.

If you choose to test a specific image then you will be asked to provide the image path. After this is  
done the program will output a simple with the model's guess as its only entry.

If you choose to use a large amount of random images then you will first be asked if you want to use the  
testing set or the validation set, the important distinction between these sets is that the testing set  
will have been used by the model when training to evaluate its own performance. The validation set however  
is untouched and therefore provides new never before seen data. After selecting one of these sets, you will  
then be asked how many images you wish to test. The images will then be tested and the data created will be  
output into a table that notes the networks confidence in its decisions, the networks guess and the actual  
status of the image. It also outputs a percentage value that represents how often it was correct.  
