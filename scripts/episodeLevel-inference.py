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
from models import BRESTModel


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
    def __init__(self, csv_file, image_folder, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.transform = transform
        valid_paths = self.metadata['path'].map(
            lambda x: os.path.exists(os.path.join(image_folder, x + '.png'))
        )
        self.metadata = self.metadata[valid_paths]
        print("After path validity filter there are {} rows of data.".format(self.metadata.shape[0]))

        # Group by ClientID and ensure each group has exactly 4 images
        self.groups = self.metadata.groupby('ClientID').filter(lambda x: len(x) == 4)

        # Print label distribution
        label_counts = self.metadata['EpisodeOutcome'].map(lambda x: 1 if x != 'N' else 0).value_counts() / 4
        print("Label distribution:", label_counts.to_dict())

        # Filter out client IDs with invalid image paths
        valid_client_ids = []
        for client_id in self.groups['ClientID'].unique():
            rows = self.groups[self.groups['ClientID'] == client_id].head(4)
            # Check if all 4 image paths exist
            valid = all(
                os.path.isfile(
                    os.path.join(image_folder, row["path"], ".png").replace("/.png", ".png")
                )
                for _, row in rows.iterrows()
            )
            if valid:
                valid_client_ids.append(client_id)

        # Map each valid ClientID to a row index in a simple DataFrame
        self.index_to_id = pd.DataFrame(valid_client_ids, columns=['ClientID'])
        self.image_folder = image_folder

    def __len__(self):
        return len(self.index_to_id)

    def __getitem__(self, idx):
        # Identify which ClientID corresponds to this index
        client_id = self.index_to_id.loc[idx, 'ClientID']

        # Select the first 4 rows for this client
        rows = self.groups[self.groups['ClientID'] == client_id].head(4)

        images = []
        labels = []

        # For each row in those 4
        for _, row in rows.iterrows():
            image_path = os.path.join(self.image_folder, row["path"], ".png")
            image_path = image_path.replace("/.png", ".png")

            # Read with plt and convert to PIL
            image_array = plt.imread(image_path)
            # Convert the numpy array to a PIL Image (scaled back to 0-255)
            pil_image = Image.fromarray((image_array * 255).astype('uint8'))
            # Ensure RGB
            image = pil_image.convert('RGB')

            # Apply optional transforms
            if self.transform:
                image = self.transform(image)

            images.append(image)

            # Label: 1 if EpisodeOutcome != 'N', else 0
            label = 1 if row['EpisodeOutcome'] != 'N' else 0
            labels.append(label)

        # Stack the 4 images into a single 4D tensor: shape [4, C, H, W]
        images = torch.stack(images)

        # Make the labels a 1D tensor with 4 elements
        labels = torch.tensor(labels, dtype=torch.long)

        return images, labels



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

#     with torch.no_grad():
#         for images, labels in tqdm(dataloader, desc="Inference Progress"):
#             images = images.to(device)
#             labels = labels.to(device)

#             outputs = model(images)
#             outputs = torch.sigmoid(outputs).squeeze(dim=1)  # shape: [batch_size]

#             all_labels.extend(labels.cpu().tolist())
#             all_probs.extend(outputs.cpu().tolist())
            
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


