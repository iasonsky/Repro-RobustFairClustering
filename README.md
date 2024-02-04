## Code for the ICLR'23 paper: [Robust Fair Clustering: A Novel Fairness Attack and Defense Framework](https://arxiv.org/pdf/2210.01953.pdf)

### Requirements:
```
python-mnist
gdown
kmedoids
pulp
torch
scikit-learn==1.2.2  # This version works.
zoopt
pyckmeans
```

### Instructions:
First of all, it might be better to download the required datasets mnist_usps and office31, cropped Yale, uncropped Yale and MTFL locally to avoid potential network connection problem. 
  - On snellius, the dataset like mnist_usps should be put in "home5/scurXXXX/.conda/envs/FACT2024/lib/python311.zip/fair_clustering/raw_data/mnist_usps".
  - On local computer, the file path is like "fair_clustering/raw_data/mnist_usps" with a Linux system.
  - The links to each dataset are:
    1. office31:
      - "domain_adaptation_features_20110616.tar.gz": "https://drive.google.com/u/0/uc?id=0B4IapRTv9pJ1WTVSd2FIcW4wRTA&export=download",
      - "office31_resnet50.zip": "https://wjdcloud.blob.core.windows.net/dataset/office31_resnet50.zip"
    2. mnist_usps:
         "https://mega.nz/folder/oHJ2UCoK#r62nRoZ0gH8NXIcgmyWReA"
    3. Extended YaleB (cropped):
         "https://academictorrents.com/details/aad8bf8e6ee5d8a3bf46c7ab5adfacdd8ad36247"
    4. Extended YaleB (uncropped)
         ""
    5. MTFL
         "https://mmlab.ie.cuhk.edu.hk/projects/TCDCN/data/MTFL.zip"
        
- For reproducing the attack section of the paper, please follow the guidance and code in the branch main of this github project.

- For reproducing the results related to defense algorithm of the paper, if you work on snellius, please navigate to the directory Fair-Clustering-Codebase:
  1. To get the CFC result in Table 3 and Table 8, use the job file run_defense.job to run Defense.py. After you test on one of the dataset, you can change to another dataset by navigating to the comment in 
     Defense.py " #Choose between Office-31, MNIST_USPS, Yale, or DIGITS", and replace the dataset name there. We didn't use a loop to experiment with all dataset in one run in order to save the computation 
     resources on snellius considering the limited provided gpu usage time units.
  2. Whichever your choice, 

- Using the code provided in these notebooks, the experimental results in the paper can be obtained.
- If there any questions, please reach out to ```{last_name}@ucdavis.edu``` where ```{last_name}``` is the first author's last name.
