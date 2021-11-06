# MRA2 -2021
An effective black-box attack methodology named malicious-function Reserved Automated Attack (MRA2) against ML-based malware detection models.

The following guide will help you install and run the project on your local machine.

Requirements
1. python
2. NVIDIA
3. Anaconda 
4. Tensorflow
5. cuDNN7 
6. CUDA

Installation steps(NIVIDIA driver has installed):

1 Install Anaconda
2 Install tensorflow
  1) Open Anaconda Prompt and enter 'conda env list' to view the current environment
  2) Create the TensorFlow environment and installing numpy:
  enter 'conda create -n tensorflow python=3.6 numpy pip'
3 Enter the TensorFlow environment
  Using conda env list, you will find that you have an additional environment calle TensorFlow.
  Enter: 'activate tensorflow' to enter the environment
4 pip install tensorflow

the tool we used: 
1) Inception-v3 that is used for visualization malware detection,
2) VirusTotal sandbox online that is used for verifing the malware AEs functions.

The programming code is uploaded with this submission to reproduce results in our paper.
