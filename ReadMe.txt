Develop Azure function app to calculate tensorflow 2 object detection inference

An Azure function app (httptrigger) has to be developed to calculate images with a given model for object detection (tensorflow 2).

Requirements concerning environment
- Python 3.7.4 has to be used
- tensorflow 2.4.1 has to be used
- use Linux for Azure function app
- You have to use your own Azure environment (no access can be given to my Auzere environment) 

On my local machine, everything is working fine on my Windows PC.

The aim is that I can deploy and run such an Azure function. Source code has to be delivered and maybe help has to be given in setting up the function app. (Actually I deployed many function apps already.) 



Description of files
====================

Program files:
TestLocal.py - main python file which is called to run this sample (python .\TestLocal.py) 
di_detect_from_image.py - doing the real job, load model, calc image
di_log.py - just to do some output (not really helpful here)

Input data:
saved_model - directory with the model
test.jpg - test image used in TestLocal.py

Output data (after running python .\TestLocal.py)):
res_img.jpg - result image after running of TestLocal.py 
res.txt - result data after running of TestLocal.py
output_console.txt - output console after run of 

General info:
pip_freeze.txt - my local installation
python_version.txt - my python version
ReadMe.txt - this file


