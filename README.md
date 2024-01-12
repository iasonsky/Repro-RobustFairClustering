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
- For the attack section of the paper, please follow the code in `Attack.ipynb`.

- For the defense section of the paper, if you work on snellius, please navigate to the directory Fair-Clustering-Codebase, using the job file run_defense.job to run Defense.py. Otherwise it's similar to attack section above.
  - Whichever your choice, it might be better to download the required datasets mnist_usps and office31 locally to avoid linkage problem.
    - On snellius, the dataset like mnist_usps should be put in /gpfs/home5/scurXXXX/.conda/envs/FACT2024/lib/python311.zip/fair_clustering/raw_data/mnist_usps.
    - The links to each dataset are:
      1. office31:
        "domain_adaptation_features_20110616.tar.gz": "https://drive.google.com/u/0/uc?id=0B4IapRTv9pJ1WTVSd2FIcW4wRTA&export=download",
        "office31_resnet50.zip": "https://wjdcloud.blob.core.windows.net/dataset/office31_resnet50.zip"
      2. mnist_usps:
         https://mega.nz/folder/oHJ2UCoK#r62nRoZ0gH8NXIcgmyWReA

- Using the code provided in these notebooks, the experimental results in the paper can be obtained.
- If there any questions, please reach out to ```{last_name}@ucdavis.edu``` where ```{last_name}``` is the first author's last name.
