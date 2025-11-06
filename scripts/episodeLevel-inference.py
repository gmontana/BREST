'''
Usage Example (from the command line):
python inference.py 
    --metadata_csv /path/to/metadata.csv 
    --image_root_dir /path/to/images/ 
    --final_csv_path /path/to/output/results.csv 
    --roc_plot_path /path/to/output/roc_curve.png 
    --model_checkpoint /path/to/model_checkpoint.pth 
    --gpu_id 0

'''
import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, roc_curve, auc

# Import your models from models.py
from models import BREST_16bit


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EpisodeDataset(Dataset):
    """
    Loads images 4 at a time for each ClientID.
    Exactly the same grouping logic you provided.
    """
    def __init__(self, csv_file, image_root, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.transform = transform
        self.image_root = image_root

        # Group by ClientID and EpisodeID and ensure each group has at least 4 images
        self.groups = self.metadata.groupby(['ClientID', 'EpisodeID'],sort=False).filter(lambda x: len(x) == 4)
        
        valid_ids = []
        cancerousCount = 0
        for ids, rows in self.groups.groupby(['ClientID', 'EpisodeID'], sort=False):
            imageFolder = self.image_root

            if rows['EpisodeOutcome'].isin(['MP', 'CIP', 'MPP', 'CIPP']).any():
                valid_rows = rows.head(4)
                cancerousCount += 1
            # if rows['EpisodeOutcome'].isin(['M', 'CI', 'B']).any():
            #     cancerousCount += 1
            #     valid_rows = rows.head(4)
            elif (rows['EpisodeOutcome'].isin(['N'])).all():
                valid_rows = rows.head(4)
            else:
                continue
            
            valid = all(os.path.isfile(os.path.join(imageFolder,row["path"] + ".png")) for _, row in valid_rows.iterrows())
            if valid:
                valid_ids.append(ids)  # add the tuple (client_id, episode_id)
        print(len(valid_ids))  
        print(cancerousCount) 
        # Create a DataFrame to map index to ClientID and EpisodeID
        self.index_to_id = pd.DataFrame(valid_ids, columns=['ClientID', 'EpisodeID'])

    def __len__(self):
        return len(self.index_to_id)

    def __getitem__(self, idx):
    # Get the ClientID and EpisodeID for this index
        client_id, episode_id = self.index_to_id.loc[idx, ['ClientID', 'EpisodeID']]

        # Get the rows for this ClientID and EpisodeID
        rows = self.groups[(self.groups['ClientID'] == client_id) & (self.groups['EpisodeID'] == episode_id)]

        if rows['EpisodeOutcome'].isin(['MP', 'CIP', 'MPP', 'CIPP']).any():
            rows = rows.head(4)
        # if rows['EpisodeOutcome'].isin(['M', 'CI', 'B']).any():
        #     rows = rows.head(4)
        elif (rows['EpisodeOutcome'].isin(['N'])).all():
            rows = rows.head(4)

        

        # Get the first 4 images for this ClientID
        images = []
        labels = []
        imageFolder = self.image_root
            
        for _, row in rows.iterrows():
            imagePath = os.path.join(imageFolder, row["path"] + ".png")
            
            np_img = self.load_16bit_png(imagePath)  # (H, W)
            # Repeat channel to get (H, W, 3)
            np_img3 = np.repeat(np_img[:, :, np.newaxis], 3, axis=2)
            # Convert to tensor, shape (3, H, W)
            tensor_img = torch.from_numpy(np_img3).permute(2, 0, 1).contiguous().float()
            if self.transform:
                tensor_img = tensor_img.unsqueeze(0)
                tensor_img = self.transform(tensor_img)
                tensor_img = tensor_img.squeeze(0)

            images.append(tensor_img)

            # Get the label and encode it as 0 or 1
            # label = 0 if (row['EpisodeOutcome'] == 'N' or row['EpisodeOutcome'] == 'B') else 1
            label = 1 if (row['EpisodeOutcome'] != 'N') else 0
            labels.append(label)

        # Stack images to create a 4D tensor and labels into a 1D tensor
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)

        return images, labels
    @staticmethod
    def load_16bit_png(path):
        """
        Loads a 16-bit PNG as float32 in [0,1].
        """
        import imageio.v3 as iio  # or import imageio if old version
        np_img = iio.imread(path)
        # Safety: handle both uint16 and int32 (from PIL)
        np_img = np_img.astype(np.float32)
        np_img /= 65535.0  # Now in [0,1]
        return np_img



def main():
    # ---------------------------------------
    # 1. Parse Command-Line Arguments
    # ---------------------------------------
    parser = argparse.ArgumentParser(description="Run model inference with selected model.")
    parser.add_argument('--metadata_csv', type=str, required=True,
                        help='Path to the metadata CSV (with columns: ClientID, path, EpisodeOutcome).')
    parser.add_argument('--image_root_dir', type=str, required=True,
                        help='Directory where .png images are stored.')
    parser.add_argument('--final_csv_path', type=str, required=True,
                        help='Output CSV path for final predictions.')
    parser.add_argument('--roc_plot_path', type=str, required=True,
                        help='Path to save the ROC curve figure (PNG).')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='Path to the model checkpoint (.pth).')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='Comma-separated list of GPU IDs to use (e.g. "0", "7", or "0,1"). Default: "0".')
    args = parser.parse_args()
  
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # ---------------------------------------
    # 2. Set Seed for Reproducibility
    # ---------------------------------------
    set_seed(66)

    # ---------------------------------------
    # 3. Prepare Dataset and Dataloader
    # ---------------------------------------
    data_transforms = transforms.Compose([
        transforms.Resize((1792, 1792)),
        transforms.ToTensor(),
        transforms.Normalize([0.0856]*3, [0.1687]*3)
    ])
    
    dataset = EpisodeDataset(
        csv_file=args.metadata_csv,
        image_folder=args.image_root_dir,
        transform=data_transforms
    )
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )

    # ---------------------------------------
    # 4. Instantiate and Load Model
    # ---------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BRESTModel()  # Or import your real model from models.py

    # If multiple GPUs are available, wrap in DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # Load checkpoint and remove 'module.' if needed
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] if k.startswith('module.') else k  # remove 'module.'
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # ---------------------------------------
    # 5. Inference
    # ---------------------------------------
    all_labels = []
    all_probs = []
            
    for images, labels in tqdm(dataloader, desc="Inference Progress"):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(images)
            outputs = torch.sigmoid(outputs)

        # Loop over each ClientID in the batch
        for idx in range(images.size(0)):
            client_probs = outputs[idx].cpu().numpy()  # probabilities for each image of the client
    #         print(client_probs)
            client_label = int(any(labels[idx].cpu().numpy()))  # true label for the client (1 if any image has cancer, else 0)
    #         print(client_label)

            all_labels.append(client_label)
            # Append the max probability for the client
            all_probs.append(np.max(client_probs))
        

    # ---------------------------------------
    # 6. ROC Curve & AUC
    # ---------------------------------------
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    print("AUC:", roc_auc)

    # Save the ROC curve without displaying
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Episode-Level)')
    plt.legend(loc="lower right")
    os.makedirs(os.path.dirname(args.roc_plot_path), exist_ok=True)
    plt.savefig(args.roc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------
    # 7. Save Results to CSV
    # ---------------------------------------
    data_dict = {
        'Episode Label': all_labels,
        'Predicted Probability': all_probs
    }
    df_out = pd.DataFrame(data_dict)
    os.makedirs(os.path.dirname(args.final_csv_path), exist_ok=True)
    df_out.to_csv(args.final_csv_path, index=False)

    print(f"Saved final CSV to: {args.final_csv_path}")
    print(f"Saved ROC curve to: {args.roc_plot_path}")


if __name__ == '__main__':
    main()


