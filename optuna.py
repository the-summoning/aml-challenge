import optuna
from optuna.pruners import MedianPruner
import yaml
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import Translator
from pathlib import Path
from main import load_data, test, info_nce_loss

def objective(trial, train_dataset, val_dataset,
              epochs: int = 10, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_layers = trial.suggest_int("n_layers", 1, 4)
    layer_choices = [512, 1024, 2048, 4096]
    hidden_layers = [trial.suggest_categorical(f"n_units_l{i}", layer_choices) for i in range(n_layers)]

    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048, 4096])
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    temp = trial.suggest_float("temp", 0.05, 0.3, log=True)
    dropout_rate = trial.suggest_categorical('dropout_rate', [0.1,0.2,0.3,0.4,0.5])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Translator(input_dim=1024, output_dim=1536, hidden_layers=hidden_layers, dropout_rate=dropout_rate)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_batch = F.normalize(y_batch, dim=-1)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = info_nce_loss(outputs, y_batch, temp=temp)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_batch = F.normalize(y_batch, dim=-1)
                
                outputs = model(X_batch)
                loss = info_nce_loss(outputs, y_batch, temp=temp)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        results = test(val_dataset, model, device)

        trial.report(results['mrr'], epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()


    return results['mrr']


def run_optuna_search(data_path: Path,
                      n_trials: int = 30, epochs: int = 10,
                      n_jobs: int = 1, sampler=None, pruner=None):
    if pruner is None:
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    train_dataset, val_dataset = load_data(data_path, config={})

    study = optuna.create_study(direction="maximize", pruner=pruner)
    func = lambda trial: objective(trial, train_dataset=train_dataset, val_dataset=val_dataset,
                                   epochs=epochs)
    study.optimize(func, n_trials=n_trials, n_jobs=n_jobs)

    print("Study statistics:")
    print("  Number of finished trials: ", len(study.trials))
    print("  Best trial:")
    trial = study.best_trial
    print("    Value: ", trial.value)
    print("    Params: ")
    for k, v in trial.params.items():
        print(f"      {k}: {v}")

    return study

if __name__ == "__main__":
    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)
    data_path = config['train_path']
    study = run_optuna_search(data_path=data_path,
                          n_trials=100, epochs=10, n_jobs=1)
    study.trials_dataframe().to_csv("optuna_trials.csv", index=False)

    best_trial_number = study.best_trial.number
    print("Best params:", study.best_params)
    print("Best trial number:", study.best_trial.number)