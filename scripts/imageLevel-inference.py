'''
Usage Example (from the command line):
python inference.py 
    --metadata_csv /path/to/metadata.csv 
    --image_root_dir /path/to/images/ 
    --final_csv_path /path/to/output/final.csv 
    --roc_plot_path /path/to/output/roc_curve.png 
    --model_checkpoint /path/to/model_checkpoint.pth 
    --model_type cad 
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
from sklearn.metrics import roc_auc_score, roc_curve

# Import your models from models.py
from models import CADModel, RiskModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CustomDataset(Dataset):
    """
    Custom dataset to load images from a directory given paths in a DataFrame.
    """
    def __init__(self, dataframe, root_dir, transform=None):
        """
        Args:
            dataframe (pandas.DataFrame): DataFrame containing 'path' and 'EpisodeOutcome'.
            root_dir (str): Directory containing images (as .png files).
            transform (callable, optional): Optional transform to be applied to each image.
        """
        # Filter out rows whose image paths do not exist
        valid_paths = dataframe['path'].map(
            lambda x: os.path.exists(os.path.join(root_dir, x + '.png'))
        )
        dataframe = dataframe[valid_paths]
        print("After path validity filter there are {} rows of data.".format(dataframe.shape[0]))

        # Prepare image paths
        self.image_paths = dataframe['path'].tolist()

        # Convert 'EpisodeOutcome' to binary labels: N -> 0, anything else -> 1
        self.labels = dataframe['EpisodeOutcome'].map(lambda x: 0 if x == 'N' else 1).tolist()
        label_counts = dataframe['EpisodeOutcome'].map(lambda x: 0 if x == 'N' else 1).value_counts() / 4
        print("Label counts:", label_counts.to_dict())

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_paths[idx] + '.png')
        # Read image using Matplotlib, then convert to PIL
        image_array = plt.imread(image_path)
        pil_image = Image.fromarray((image_array * 255).astype('uint8'))

        # Convert grayscale to RGB
        image = pil_image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


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
    parser.add_argument('--model_type', type=str, required=True, choices=['cad', 'risk'],
                        help="Which model to run: 'cad' or 'risk'.")
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='Comma-separated list of GPU IDs to use (e.g. "0", "7"). Default: "0".')
    args = parser.parse_args()
  
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # ---------------------------------------
    # 2. Set Seed for Reproducibility
    # ---------------------------------------
    set_seed(66)

    # ---------------------------------------
    # 3. Prepare Dataset and Dataloader
    # ---------------------------------------
    df = pd.read_csv(args.metadata_csv)
    data_transforms = transforms.Compose([
        transforms.Resize((1792, 1792)),
        transforms.ToTensor(),
        transforms.Normalize([0.0856]*3, [0.1687]*3)
    ])
    val_dataset = CustomDataset(df, args.image_root_dir, data_transforms)
    val_loader = DataLoader(val_dataset, batch_size=4, num_workers=2, shuffle=False)

    # ---------------------------------------
    # 4. Choose & Load Model
    # ---------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select the model class based on user input
    if args.model_type.lower() == "cad":
        model = CADModel()
        print("Using CADModel...")
    else:
        model = RiskModel()
        print("Using RiskModel...")

    # If multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # Load checkpoint
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    model.to(device)
    model.eval()

    # ---------------------------------------
    # 5. Inference
    # ---------------------------------------
    client_true_labels = []
    client_predicted_probs = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Inference Progress"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            outputs = torch.sigmoid(outputs)

            for idx in range(images.size(0)):
                client_probs = outputs[idx].cpu().numpy()
                client_label = labels[idx].cpu().numpy()
                client_true_labels.append(client_label)
                client_predicted_probs.append(client_probs)

    # ---------------------------------------
    # 6. Post-Processing & Save Results
    # ---------------------------------------
    test_df = pd.read_csv(args.metadata_csv)
    # Convert 'EpisodeOutcome' to a single label for each client
    labels_grouped = test_df.groupby('ClientID')['EpisodeOutcome'] \
                            .transform(lambda x: 0 if (x == 'N').all() else 1)
    test_df['label'] = labels_grouped

    predicted_probs_df = pd.DataFrame(client_predicted_probs, columns=["Predicted_Probabilities"])
    predicted_probs_df["ClientID"] = test_df["ClientID"]

    unique_clients = test_df[['ClientID', 'label']].drop_duplicates()
    max_probs = predicted_probs_df.groupby('ClientID')['Predicted_Probabilities'].max().reset_index()
    merged_df = pd.merge(unique_clients, max_probs, on='ClientID', how='inner')

    y_true = merged_df['label']
    y_scores = merged_df['Predicted_Probabilities']

    roc_auc = roc_auc_score(y_true, y_scores)
    print(f"ROC-AUC Score: {roc_auc:.3f}")

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, lw=lw, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    os.makedirs(os.path.dirname(args.roc_plot_path), exist_ok=True)
    plt.savefig(args.roc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    final_data = {
        'Client True Labels': merged_df['label'],
        'Probabilities': merged_df['Predicted_Probabilities']
    }
    final_df = pd.DataFrame(final_data)
    os.makedirs(os.path.dirname(args.final_csv_path), exist_ok=True)
    final_df.to_csv(args.final_csv_path, index=False)

    print(f"Final CSV saved to: {args.final_csv_path}")
    print(f"ROC curve saved to: {args.roc_plot_path}")


if __name__ == "__main__":
    main()
