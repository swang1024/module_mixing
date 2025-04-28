In Appendix, we provided extra details and experimental results for better understanding. The Appendix is organized as below:

A. EFFICIENT FEATURE TRANSFORMATION

B. ALGORITHM

C. DATASET AND IMPLEMENTATION DETAILS

D. ADDITIONAL EXPERIMENTAL RESULTS

We provide the steps for experiments on Office-31 dataset.

Step 1: Prepare dataset

Please manually download Office-31 dataset from [the official website](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA)
and unzip the dataset to folder "./data" and sort the data like below. We provide the image_unida_list.txt files for 
each domain.

```
./data
├── Office
│   ├── Amazon
|       ├── ...
│       ├── image_unida_list.txt
│   ├── Dslr
|       ├── ...
│       ├── image_unida_list.txt
│   ├── Webcam
|       ├── ...
│       ├── image_unida_list.txt
├── OfficeHome
│   ├── ...
├── VisDA
│   ├── ...
```
Step 2: Train Independent models (ResNet18 with EFT modules) for each domain

-- first download pretrained resnet18 model on ImageNet and save it as 'resnet18.pth'

-- then run 
    python train_encoder_office_eft.py

Step 3: Build a layerwise module-mixing model by transfering from source domains "amazon" and "webcam' to "dslr" with their source models 

-- run

python module_mixing_office_distance_correlation.py
