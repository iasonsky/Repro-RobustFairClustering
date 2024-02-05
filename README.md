## Reproducibility study for the ICLR'23 paper: [Robust Fair Clustering: A Novel Fairness Attack and Defense Framework](https://arxiv.org/pdf/2210.01953.pdf)

### Environment and requirements:
To install the required packages, create a new conda environment using the provided YAML file:
```bash
conda env create -f FACT2024.yml
```
Activate the environment:
 ```bash
conda activate FACT2024
```

Install the requirements:
```bash
pip install -r requirements.txt
```

In order to run the KFC algorithm `IBM-CPLEX 20.1.0` needs to be installed. Instructions [here](https://www.ibm.com/support/pages/downloading-ibm-ilog-cplex-optimization-studio-2010). 

Once you have succesfuly installed it you need to run:  
```
python "C:\Program Files\IBM\ILOG\CPLEX_Studio201\python\setup.py" install
``` 

And then: 
```
conda install -c ibmdecisionoptimization cplex
```
And now you can succesfully run cplex directly from python using PuLP!
### Datasets:
First of all, it might be better to download the required datasets mnist_usps and office31, cropped Yale, uncropped Yale and MTFL locally to avoid potential network connection problems. 
  - On snellius, the dataset like mnist_usps should be put in `home5/scur1047/.conda/envs/FACT2024/lib/python311.zip/fair_clustering/raw_data/mnist_usps`.
  - If you run the experiments locally find the `.conda` directory and put the dataset `<user>/.conda/envs/FACT2024/lib/<python_version>.zip/fair_clustering/raw_data/mnist_usps`.
  - The links to each dataset are:
    1. `Office-31`: put them in `home5/scurXXXX/.conda/envs/FACT2024/lib/python311.zip/fair_clustering/raw_data/office31`. The links are: 
          - [`domain_adaptation_features_20110616.tar.gz`](https://drive.google.com/u/0/uc?id=0B4IapRTv9pJ1WTVSd2FIcW4wRTA&export=download)
          - [`office31_resnet50.zip`](https://wjdcloud.blob.core.windows.net/dataset/office31_resnet50.zip)
    2. `MNIST_USPS`[[link](https://mega.nz/folder/oHJ2UCoK#r62nRoZ0gH8NXIcgmyWReA)]: put it in `home5/scurXXXX/.conda/envs/FACT2024/lib/python311.zip/fair_clustering/raw_data/mnist_usps`.
    3. `Extended YaleB (cropped)` [[link](https://academictorrents.com/details/aad8bf8e6ee5d8a3bf46c7ab5adfacdd8ad36247)]: put it in `home5/scurXXXX/.conda/envs/FACT2024/lib/python311.zip/fair_clustering/raw_data/extended_yaleB`.
    4. `Extended YaleB (uncropped)` [[link](https://drive.google.com/file/d/1NSzt-Ld_HMrQKw_zrplpZlLbR0EaUVcJ/view?usp=sharing)]: put it in `home5/scurXXXX/.conda/envs/FACT2024/lib/python311.zip/fair_clustering/raw_data/extended_yaleB_alter`.
    5. `MTFL` [[link](https://mmlab.ie.cuhk.edu.hk/projects/TCDCN/data/MTFL.zip)]: put it in `home5/scurXXXX/.conda/envs/FACT2024/lib/python311.zip/fair_clustering/raw_data/mtfl`.
        
### Attack
In order to reproduce the attack results of the original paper navigate to Fair-Clustering-Codebase and run the following:
```
python attack.py --dataset_name <DATASET_NAME> --cl_algo <CLUSTERING_ALGORITHM>
```

To run the attacks including the extra metrics:
```
python attack_extra_metrics.py --dataset_name <DATASET_NAME> --cl_algo <CLUSTERING_ALGORITHM>
```

To run the attacks for the plots of Figure 1:
```
python attack_for_plots.py --dataset_name <DATASET_NAME> --cl_algo <CLUSTERING_ALGORITHM>
```

In order to get the best configuration for the new combined attack:
```
python attack_extra_metrics_ng_grid.py --dataset_name <DATASET_NAME> --cl_algo <CLUSTERING_ALGORITHM>
```

Finally to get the results for the new `combined_attack` or `attack_min_cluster_ratio`
```
python attack_extra_metrics_ng.py --dataset_name <DATASET_NAME> --cl_algo <CLUSTERING_ALGORITHM> --attack combined
python attack_extra_metrics_ng.py --dataset_name <DATASET_NAME> --cl_algo <CLUSTERING_ALGORITHM> --attack min_cluster_ratio
```

Note: We have only run a small grid search and the provided best `combined_attack` configuration is for Office-31 and SFD. 


If you want to run the experiments on Snellius you can run the `run_attack.job` file and adjust the file you want to run each time, as well as the directories to save your `.out` files.

#### Results:
All the results for pre-attack, post-attack and random attack for all datasets are presented in `Results_Office_MNIST.ipynb`, `Results_DIGITS_Yale.ipynb` and `Results_MTFL_Yale_alter.ipynb`. The results of Figure 1 can be found in `plot_data/repro_figure_2.ipynb`

### Defense
- For reproducing the results related to defense algorithm of the paper, if you work on snellius, please navigate to the directory home5/scur1047/Defense/CFC-master/Fair-Clustering-Codebase:
  1. To get the CFC result in Table 3 and Table 8, use the job file `run_defense.job` to run `Defense.py`. After you test on one of the dataset, you can change to another dataset by navigating to the comment in 
     `Defense.py` " #Choose between Office-31, MNIST_USPS, Yale, or DIGITS", and replace the dataset name there. We didn't use a loop to experiment with all dataset in one go, in order to save the computation 
     resources on snellius considering the limited gpu time units.
  2. To get Figure 3 in Appendix, use the job file `run_defense_multi.job` to run `Defense_multi.py`. By gathering results from experiments on each of the 4 datasets Office-31, MNIST_USPS, Yale, or DIGITS, use the   
     results data to run `Fair-Clustering-Codebase/plot_data/repro_figure_6.ipynb`
  3. To get the CFC result in Table 11 (to test if CFC algorithm has been robust to different new attacks), use the job file `run_extra_eval.job` to run `Defense_add_eval.py`. By searching and navigating to the "obj 
     = Objective", you can change the type of attack there. i.e. In obj = Objective(attack_min_cluster_ratio, dim), the attack type is "attack_min_cluster_ratio", you can change it to "combined_attack" or 
     "attack_balance" or "attack_entropy". The results of extra evaluation metrics for CFC will also be included in this experiment.
  4. To test CFC on different datasets like MTFL and uncropped Yale, you can still use the job file `run_extra_eval.job` to run `Defense_add_eval.py`. After you test on one of the dataset, you can change to another  
     dataset by navigating to the comment in `Defense_add_eval.py` " #Choose between Office-31, MNIST_USPS, Yale, or DIGITS", and replace the dataset name there with "MTFL" or "Yale_alter". 

- With the guideline and the code provided in these notebooks, the experimental results in the paper can be obtained without extra efforts.
- If there any questions, please reach out to `iason.skylitsis@student.uva.nl` for the attack parts and ```z3.feng@student.vu.nl``` for the defense parts. We'll gladly help you solve any issues you may encounter.

