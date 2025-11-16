import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split
from pathlib import Path

def get_data(data_path: Path):
    data = np.load(data_path)
    caption_embeddings = data['captions/embeddings']
    image_embeddings = data['images/embeddings']
    caption_labels = data['captions/label']
    data.close()

    X, y = torch.tensor(caption_embeddings), torch.tensor(image_embeddings[np.argmax(caption_labels, axis=1)])

    return X, y

def get_datasets(X_abs, y_abs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    print('Texts shape', X_abs.shape)
    print('Images shape', y_abs.shape)

    dataset = TensorDataset(X_abs, y_abs)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

    return train_dataset, val_dataset