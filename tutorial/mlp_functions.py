
import pandas as pd
from mlmnemonist import ExperimentRunner
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np

def load_raw_data(runner: ExperimentRunner):
    train_name = runner.cfg.DATASET.TRAIN_NAME
    test_name = runner.cfg.DATASET.TEST_NAME
    runner.train_df = pd.read_csv(runner.reveal_true_path(train_name))
    runner.test_df = pd.read_csv(runner.reveal_true_path(test_name))

def process_data(runner: ExperimentRunner):
    runner.train_X = runner.train_df.drop(columns='median_house_value').to_numpy()
    runner.train_Y = runner.train_df['median_house_value'].to_numpy()
    runner.test_X = runner.test_df.drop(columns='median_house_value').to_numpy()
    runner.test_Y = runner.test_df['median_house_value'].to_numpy()



class MyMLP(nn.Module):
    """
    A self explanatory MLP model with ReLU activations and 2 hidden layers
    """
    def __init__(self, input_features: int, hidden_layer_1: int, hidden_layer_2: int):
        super(MyMLP, self).__init__()
        self.l1 = nn.Linear(input_features, hidden_layer_1)
        self.l2 = nn.Linear(hidden_layer_1, hidden_layer_2)
        self.l3 = nn.Linear(hidden_layer_2, 1)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        return x


def setup_device(runner: ExperimentRunner):
    # extract the device from config
    device = runner.cfg.SOLVER.DEVICE
    if device == 'cpu':
        runner.device = torch.device('cpu')
    elif device == 'cuda' and torch.cuda.is_available():
        runner.device = torch.device('cuda')
    else:
        raise NotImplementedError(f"device {device} is not implemented!")


def setup_model(runner: ExperimentRunner):
    # extract the method from config
    method = runner.cfg.SOLVER.METHOD
    # construct the model from registry
    cfg_h_params = runner.cfg.MODEL.HYPER_PARAMETERS
    my_model = MyMLP(input_features=cfg_h_params.IN_FEATURES,
                     hidden_layer_1=cfg_h_params.H1,
                     hidden_layer_2=cfg_h_params.H2)
    # set the function in the cache to save weights
    my_model = runner.CACHE.SET_M('mlp-key', my_model)
    my_model.to(runner.device)
    if runner.verbose > 0:
        print("Model state dict")
        print(my_model)


def setup_training(runner: ExperimentRunner):
    my_model = runner.CACHE.GET_M('mlp-key').to(runner.device)
    runner.criterion = nn.MSELoss()
    if runner.cfg.SOLVER.OPTIMIZER_TYPE == 'adam':
        runner.optim = torch.optim.Adam(my_model.parameters(), lr=runner.cfg.SOLVER.LR)
    else:
        raise NotImplementedError(f"Optimizer type not implemented"
                                  f"{runner.cfg.SOLVER.OPTIMIZER_TYPE}")




def my_custom_run(runner: ExperimentRunner, show_freq=50):
    # Get model from cache
    my_model = runner.CACHE.GET_M('mlp-key')

    # Get the epoch number and history from cache
    # if it is not cached from before set it to zero
    epoch_i = runner.CACHE.SET_IFN('epoch_i', 0)
    loss_history = runner.CACHE.SET_IFN('loss-history', {'train': [], 'test': []})
    for epoch_i in range(epoch_i, 400):

        # Shuffle all the indices first
        inds = np.arange(runner.train_X.shape[0])
        np.random.shuffle(inds)

        batch_size = runner.cfg.SOLVER.BATCH_SIZE

        loss_train = []
        my_model.train()

        for L in range(0, runner.train_X.shape[0], batch_size): 
            R = min(L + batch_size, runner.train_X.shape[0])
            # Find the range [L:R]
            X = torch.from_numpy(runner.train_X[inds[L:R],:]).float().to(runner.device)
            y = torch.from_numpy(runner.train_Y[inds[L:R]]).float().to(runner.device)
            loss = runner.criterion(my_model(X).squeeze(), y)
            # Find the loss between the predicted batch and y
            loss.backward()
            runner.optim.step()
            loss_train.append(loss.detach().cpu().item())

        # Find the average loss and add it to the trainign loss history
        mean_loss = sum(loss_train) / len(loss_train)
        loss_history['train'].append(mean_loss)

        # Now we will check the loss on the whole test dataset
        loss_test = []
        # Shuffle all the indices first
        inds = np.arange(runner.test_X.shape[0])
        np.random.shuffle(inds)
        my_model.eval()
        with torch.no_grad():
            for L in range(0, runner.test_X.shape[0], batch_size):
                R = min(L + batch_size, runner.test_X.shape[0])
                X = torch.from_numpy(runner.test_X[inds[L:R], :]).float().to(runner.device)
                y = torch.from_numpy(runner.test_Y[inds[L:R]]).float().to(runner.device)
                loss = runner.criterion(my_model(X).squeeze(), y)
                loss_test.append(loss.detach().cpu().item())
        # Find the average loss and add them to loss history
        mean_loss = sum(loss_test) / len(loss_test)
        loss_history['test'].append(mean_loss)


        # Display the losses
        if (epoch_i + 1) % show_freq == 0:
            clear_output()
            plt.plot(list(range(len(loss_history['train']))), loss_history['train'], label='loss-train')
            plt.plot(list(range(len(loss_history['test']))), loss_history['test'], label='loss-test')
            plt.legend()
            plt.show()

        # Caching and saving checkpoints
        runner.CACHE.SET('epoch_i', epoch_i)
        runner.CACHE.SET('loss-history', loss_history)
        runner.CACHE.SET_M('mlp-key', my_model)
        runner.CACHE.SAVE()

    return loss_history['test'][-1]