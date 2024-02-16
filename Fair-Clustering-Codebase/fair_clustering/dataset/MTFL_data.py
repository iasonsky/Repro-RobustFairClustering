import os
import sys
import numpy as np
from PIL import Image
import pandas as pd
import urllib.request
import zipfile
from fair_clustering.dataset import ImageDataset

class MTFL(ImageDataset):
    url = "https://mmlab.ie.cuhk.edu.hk/projects/TCDCN/data/MTFL.zip"
    filename = "MTFL.zip"

    target_folder = "/home/<user_name>/.conda/envs/RobustFairClustering/lib/python311.zip/fair_clustering/raw_data/mtfl/MTFL"

    #script_directory = os.path.dirname(os.path.abspath(__file__))
    #print(f"s", script_directory)
    #target_folder_local = os.path.join(script_directory, "..", "fair_clustering", "raw_data", "mtfl")

    # Download the file
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)

    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

        # Unzip the file into the target folder
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(target_folder)

    dataset_name = "MTFL"
    dataset_dir = os.path.join(sys.path[1], os.path.join("fair_clustering", "raw_data", 'mtfl'))
    def __init__(self, center=True):
        columns = ['image_path', 'x1', 'x2', 'x3', 'x4', 'x5', 'y1', 'y2', 'y3', 'y4', 'y5', 'gender', 'smile', 'glasses', 'head_pose']

        data = pd.DataFrame(columns=columns)
        dataset_dir = os.path.join(self.dataset_dir, "MTFL")
        with open(os.path.join(dataset_dir, 'training.txt'), 'r') as file:
            for line in file:
                try:
                    fields = line.split()
                    row = pd.Series(fields, index=columns)
                    data = pd.concat([data, pd.DataFrame([row])], ignore_index=True)
                except:
                    print(f"Skipping line due to error: {line}")

            data[['x1', 'x2', 'x3', 'x4', 'x5', 'y1', 'y2', 'y3', 'y4', 'y5']] = data[['x1', 'x2', 'x3', 'x4', 'x5', 'y1', 'y2', 'y3', 'y4', 'y5']].apply(pd.to_numeric)
            data[['gender', 'smile', 'glasses', 'head_pose']] = data[['gender', 'smile', 'glasses', 'head_pose']].astype('category')

        training_data = data
        glasses = training_data[training_data['glasses'] == '1'].sample(1000)
        no_glasses = training_data[training_data['glasses'] == '2'].sample(1000)

        balanced_data = pd.concat([glasses, no_glasses])
        balanced_data['gender'] = balanced_data['gender'].astype(int) - 1
        balanced_data['smile'] = balanced_data['smile'].astype(int) - 1
        balanced_data['glasses'] = balanced_data['glasses'].astype(int) - 1
        all_img_paths = balanced_data['image_path'].tolist()

        X = [np.array(Image.open(os.path.join(dataset_dir, p.replace("\\", os.path.sep))).resize((42,48))) for p in all_img_paths]
        X = np.asarray(X)
        X = np.reshape(X, (X.shape[0], -1))
        y = balanced_data.drop(['smile', 'head_pose', 'glasses', 'image_path', 'x1', 'x2', 'x3', 'x4', 'x5', 'y1', 'y2', 'y3', 'y4', 'y5'], axis=1).values
        y = np.asarray(y).reshape(-1)
        y = y.astype(int)
        s = balanced_data['glasses'].values
        s = np.asarray(s)

        super(MTFL, self).__init__(
            X=X,
            y=y,
            s=s,
            center=center,
        )


if __name__ == "__main__":
    dataset = MTFL()
    X, y, s = dataset.data
    stat = dataset.stat