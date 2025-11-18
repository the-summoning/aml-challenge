This project contains two different approaches, each implemented in its own Jupyter notebook.

To train and test any single model, simply run all cells in its notebook:

first_approach.ipynb or second_approach.ipynb (first approach is the one that has achieved an higher score on the public leaderboard)

If you prefer to skip training and use an already saved checkpoint for testing, just comment out the training line inside the notebook:

`train_model(model, save_path, train_dataset, val_dataset, batch_size, epochs, lr, patience, queue_size, weight_decay)`
