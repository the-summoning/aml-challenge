
import numpy as np
import torch
from model import Translator
import torch.nn.functional as F
from eval import evaluate_retrieval
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from tqdm import tqdm


def pad_and_standardize(data: np.array, pad: bool, pad_val: int) -> torch.Tensor:
    data_torch = torch.from_numpy(data).float()
    if pad:
        data_torch = F.pad(data_torch, (0, pad_val), mode="constant", value=0)

    mean = data_torch.mean(dim=0, keepdim=True)
    std = data_torch.std(dim=0, keepdim=True) + 1e-8
    data_standardized = (data_torch - mean) / std

    return data_standardized


def preprocess(X_abs: np.array, Y_abs: np.array, pad: bool, normalize: bool=True) -> tuple[torch.Tensor, torch.Tensor]:
    assert X_abs.ndim == 2 and Y_abs.ndim == 2, "Both data must be 2D"

    x_pad = max(Y_abs.shape[1] - X_abs.shape[1], 0)
    y_pad = max(X_abs.shape[1] - Y_abs.shape[1], 0)

    X_pre = pad_and_standardize(X_abs, pad, x_pad)
    Y_pre = pad_and_standardize(Y_abs, pad, y_pad)

    if normalize:
        X_pre = F.normalize(X_pre, dim=1)
        Y_pre = F.normalize(Y_pre, dim=1)

    return X_pre, Y_pre


def train_model(model_path: Path, mode: str, 
                train_loader: DataLoader, val_loader: DataLoader,
                pad: bool, epochs: int, lr: float) -> Translator:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    model = Translator(pad=pad,mode=mode).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()

        train_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = 1 - F.cosine_similarity(outputs, y_batch, dim=1).mean()
            #loss = F.mse_loss(outputs, y_batch)

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

                loss = 1 - F.cosine_similarity(outputs, y_batch, dim=1).mean()
                #loss = F.mse_loss(outputs, y_batch)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            Path(model_path).parent.mkdir(parents=True, exist_ok=True)

            torch.save(model.state_dict(), model_path)

            print(f"✓ Saved best model (val_loss={val_loss:.6f})")

    return model

def eval_on_val(X_val: np.ndarray, y_val: np.ndarray, pad: bool, 
                normalize: bool, model = None, mode: str ='affine',
                model_path: Path = None) -> dict:
    gt_indices = torch.arange(len(y_val))
    
    X, y = preprocess(X_val, y_val, pad, normalize)

    if model_path:
        model = Translator(pad=pad, mode=mode)

        state = torch.load(model_path)
        model.load_state_dict(state)
        
    model.eval()

    with torch.inference_mode():
        translated = model(X)

    results = evaluate_retrieval(translated, y, gt_indices)
    
    return results

if __name__ == "__main__":
    batch_size = 512
    epochs = 100
    lr = 0.0005

    data = np.load(Path('data/train/train.npz'))
    caption_embeddings = data['captions/embeddings']
    image_embeddings = data['images/embeddings']
    caption_labels = data['captions/label']

    X_abs = caption_embeddings # captions space
    y_abs = image_embeddings[np.argmax(caption_labels, axis=1)] # images space

    X, y = preprocess(X_abs, y_abs, pad=False, normalize=False)
    

    n_train = int(0.9 * len(X))
    train_split = torch.zeros(len(X), dtype=torch.bool)
    train_split[:n_train] = 1

    X_train, X_val = X[train_split], X[~train_split]
    y_train, y_val = y[train_split], y[~train_split]

    print(X_train.shape, X_val.shape)
    print(y_train.shape, y_val.shape)


    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)


    model = train_model('models/exp1.pth', 'affine', train_loader, val_loader, False, epochs, lr)

    results = eval_on_val(X_val.numpy(), y_val.numpy(), pad=False, normalize=False, mode='affine', model='models/exp1.pth')
    print("Test Results:", results)


