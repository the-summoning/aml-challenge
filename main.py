
from typing import Literal
import numpy as np
import torch
from model import Translator
import torch.nn.functional as F
from eval import generate_submission, eval_on_val
from torch.utils.data import TensorDataset, DataLoader
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
                train_loader: DataLoader, val_loader: DataLoader,
                epochs: int, lr: float) -> Translator:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

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

            Path(model_path).parent.mkdir(parents=True, exist_ok=True)

            torch.save(model.state_dict(), model_path)

            print(f"✓ Saved best model (val_loss={val_loss:.6f})")

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

def load_data(data_path: Path, config: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    use_pad = config['pad']
    normalize = config['normalize'] 
    standardize = config['standardize']

    data = np.load(data_path)
    caption_embeddings = data['captions/embeddings']
    image_embeddings = data['images/embeddings']
    caption_labels = data['captions/label']

    X_abs, y_abs = preprocess(caption_embeddings, image_embeddings[np.argmax(caption_labels, axis=1)], 
                              pad=use_pad, standardize=standardize, normalize=normalize)
    
    print('Texts shape', X_abs.shape)
    print('Images shape', X_abs.shape)

    n_train = int(0.9 * X_abs.shape[0])
    train_split = torch.zeros(X_abs.shape[0], dtype=torch.bool)
    train_split[:n_train] = 1
    
    X_train, X_val = X_abs[train_split], X_abs[~train_split]
    y_train, y_val = y_abs[train_split], y_abs[~train_split]
    
    return X_train, y_train, X_val, y_val

    
def test(model: Translator, X_val: torch.Tensor, y_val: torch.tensor, device):
    results = eval_on_val(X_val, y_val, model=model, device=device)
    print("Test Results:", results)


def train(config: dict, model: Translator, X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor):
    
    batch_size = config['batch_size']
    epochs = config['num_epochs']
    lr = config['learning_rate']
    
    model_save_path = config['model_save_path']
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    train_model(model, model_save_path, 'affine', train_loader, val_loader, epochs, lr)

    print('Finished training. Now testing using best model...')


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
    X_train, Y_train, X_val, y_val = load_data(data_path, config)
    extract_anchors_method = config['anchors_method']
    extract_anchors_number = config['anchors_num']
    X_anchors = extract_anchors(X_train, extract_anchors_method, extract_anchors_number).to(device) if use_relative else None
    model_args = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'mode': mode,
        'use_relative': use_relative,
        'anchors': X_anchors
    }
    model = Translator(**model_args).to(device)

    train(config=config, model=model, X_train=X_train, y_train=Y_train, X_val=X_val, y_val=y_val)

    test_path = config['test_path']
    model_save_path = config['model_save_path']
    state = torch.load(model_save_path)
    model.load_state_dict(state)

    test(model, X_val, y_val, test_path, device)
    generate_submission(model, Path(test_path), device=device)



