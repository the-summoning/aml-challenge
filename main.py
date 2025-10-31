
from typing import Literal
import numpy as np
import torch
from model import Translator
import torch.nn.functional as F
from eval import generate_submission, eval_on_val
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import yaml


def pad(data: torch.Tensor, pad_val: int) -> torch.Tensor:
    return F.pad(data, (0, pad_val), mode="constant", value=0)

def standardize(data: torch.Tensor) -> torch.Tensor:

    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True) + 1e-8
    data_standardized = (data - mean) / std

    return data_standardized

def preprocess(X_abs: np.array, Y_abs: np.array, pad: bool, standardize: bool, normalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
    assert X_abs.ndim == 2 and Y_abs.ndim == 2, "Both data must be 2D"
    X_abs, Y_abs = torch.from_numpy(X_abs).float(), torch.from_numpy(Y_abs).float()

    # if pad:
    #     x_pad = max(Y_abs.shape[1] - X_abs.shape[1], 0)
    #     y_pad = max(X_abs.shape[1] - Y_abs.shape[1], 0)

    #     X_abs = pad(X_abs, x_pad)
    #     Y_abs = pad(Y_abs, y_pad)

    if standardize:
        X_abs = standardize(X_abs)
        Y_abs = standardize(Y_abs)

    if normalize:
        X_abs = F.normalize(X_abs, dim=1)
        Y_abs = F.normalize(Y_abs, dim=1)

    return X_abs, Y_abs


def train_model(model: Translator, model_path: Path, mode: str, 
                train_dataset: TensorDataset, val_dataset: TensorDataset, batch_size: int,
                epochs: int, lr: float, patience: int) -> Translator:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss = float('inf')
    no_improvements = 0


    for epoch in range(epochs):
        model.train()

        train_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)

            #loss = 1 - F.cosine_similarity(outputs, y_batch, dim=1).mean()
            loss = F.mse_loss(outputs, y_batch)

            loss.backward()

            optimizer.step()

            if mode == 'isometry':
                model.orthogonalize()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()

        val_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)

                #loss = 1 - F.cosine_similarity(outputs, y_batch, dim=1).mean()
                loss = F.mse_loss(outputs, y_batch)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvements = 0

            Path(model_path).parent.mkdir(parents=True, exist_ok=True)

            torch.save(model.state_dict(), model_path)

            print(f"✓ Saved best model (val_loss={val_loss:.6f})")
        elif no_improvements >= patience:
            return model
        else:
            no_improvements += 1

    return model

def extract_anchors(data: torch.Tensor, method: Literal['pca', 'k-means', 'random'], anchors_number: int):
    assert isinstance(data, torch.Tensor) and data.ndim == 2 and data.shape[0] > 0, "Expected a valid tensor"
    assert method in ['pca', 'k-means', 'random'], f'Method {method} not supported'
    assert isinstance(anchors_number, int) and anchors_number > 0, "Expected a natural positive number"

    data_np = data.cpu().numpy()

    if method == 'pca':
        # PCA already returns normalized anchors
        pca = PCA(n_components=anchors_number)
        pca.fit(data_np)
        
        anchors = torch.from_numpy(pca.components_).float()
    elif method == 'k-means':
        kmeans = KMeans(n_clusters=anchors_number, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(data_np)
        
        anchors = torch.from_numpy(kmeans.cluster_centers_).float()
    else:
        anchors = data[torch.randperm(data.size(0))[:anchors_number]]

    return anchors

def load_data(data_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    data = np.load(data_path)
    caption_embeddings = data['captions/embeddings']
    image_embeddings = data['images/embeddings']
    caption_labels = data['captions/label']

    X_abs, y_abs = torch.tensor(caption_embeddings), torch.tensor(image_embeddings[np.argmax(caption_labels, axis=1)])
    
    print('Texts shape', X_abs.shape)
    print('Images shape', X_abs.shape)
    
    dataset = TensorDataset(X_abs, y_abs)
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(42))
    
    return train_dataset, val_dataset

    
def test(val_dataset: TensorDataset, model: Translator, device):
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
    for x_val, y_val in val_loader:
        results = eval_on_val(x_val, y_val, model=model, device=device)
    return results


if __name__ == "__main__":
    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = config['input_dim']
    output_dim = config['output_dim']
    anchors_number = config['anchors_num']
    use_relative = config['use_relative']
    mode = config['model_mode']
    
    data_path = config['data_path']
    hidden_layers = config['hidden_layers']
    model_save_path = config['model_save_path']
    batch_size = config['batch_size']
    epochs = config['num_epochs']
    lr = config['learning_rate']
    temp = config['temperature']
    patience = config['patience']
    test_path = config['test_data_path']

    train_dataset, val_dataset = load_data(data_path, dict())
    #X_anchors = extract_anchors(X_train, extract_anchors_method, extract_anchors_number).to(device) if use_relative else None

    model_args = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_layers': hidden_layers,
    }
    model = Translator(**model_args).to(device)

    train_model(model, model_save_path, train_dataset, val_dataset, batch_size, epochs, lr, temp, patience)

    print('Finished training. Now testing using best model...')

    state = torch.load(model_save_path)
    model.load_state_dict(state)
    results = test(val_dataset, model, device)
    print("Test Results:", results)


    generate_submission(model, Path(test_path), device=device)


