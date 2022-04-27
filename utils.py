import bs4
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from tqdm.notebook import trange

from pathlib import Path
import secrets
from typing import NamedTuple

CLASSES = ["hound", "spaniel", "terrier", "retriever", "poodle", "setter", "schnauzer",
"mountain", "bulldog", "sheepdog"]

DATA_PATH = Path("data/")
CHECKPOINT_PATH = Path("checkpoints/")
RESULTS_PATH = Path("results/")

def save_dataset(data_x, data_y, type, size):
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    tensor_x = torch.from_numpy(data_x).float()
    tensor_x = tensor_x.permute(0, 3, 1, 2).contiguous()
    tensor_y = torch.from_numpy(data_y).long()

    save_tensor(tensor_x, tensor_y, type, size)

def load_dataset(type, size):
    tensor_x, tensor_y = load_tensor(type, size)
    nparray_x = tensor_x.round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    nparray_y = tensor_y.numpy()

    return (nparray_x, nparray_y)

def load_tensor(type, size, device=torch.device("cpu")):
    tensor_x = torch.load(DATA_PATH / f"{type}_data_{size}.pt")
    tensor_y = torch.load(DATA_PATH / f"{type}_label_{size}.pt")
    assert tensor_x.is_contiguous()
    assert tensor_y.is_contiguous()
    return (tensor_x.to(device), tensor_y.to(device))

def save_tensor(tensor_x, tensor_y, type, size):
    assert tensor_x.is_contiguous()
    assert tensor_y.is_contiguous()
    torch.save(tensor_x, DATA_PATH / f"{type}_data_{size}.pt")
    torch.save(tensor_y, DATA_PATH / f"{type}_label_{size}.pt")

def load_results(name, type, size):
    return pd.read_csv(RESULTS_PATH / f"{name}_{type}_{size}.csv")

def save_results(df, name, type, size):
    df.to_csv(RESULTS_PATH / f"{name}_{type}_{size}.csv")

def load_checkpoint(net, id):
    net.load_state_dict(torch.load(CHECKPOINT_PATH / f"checkpoint_{id}.pt", map_location=torch.device('cpu')))

def save_checkpoint(net, id):
    torch.save(net.state_dict(), CHECKPOINT_PATH / f"checkpoint_{id}.pt")

### Utility functions ###

def render_2d(tensor):
    assert tensor.dim() == 3 and tensor.size(0) == 3
    plt.imshow(np.transpose(tensor.numpy() , (1, 2, 0)))
    plt.show()

def count_num_params(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def get_accuracy(preds, labels):
    bs = preds.size(0)
    num_matches = (preds == labels).sum()
    return num_matches.detach().item() / bs

### Experiment functions ###

def eval_test_accuracy(net, mean, std, test_sets, input_size, batch_size=200):
    test_x, test_y = test_sets

    running_accuracy = 0
    num_batches = test_x.size(0) // batch_size

    pred_y = []
    true_y = []

    for i in range(0, num_batches*batch_size, batch_size):
        batch_x = (test_x[i:i+batch_size] - mean) / std
        batch_y = test_y[i:i+batch_size]

        inputs = batch_x.view((batch_size,) + input_size)
        scores = net(inputs)

        preds = scores.argmax(dim=1)
        running_accuracy += get_accuracy(preds, batch_y)

        pred_y.append(preds.cpu().numpy())
        true_y.append(batch_y.cpu().numpy())

    total_accuracy = running_accuracy / num_batches
    return (total_accuracy, np.concatenate(pred_y), np.concatenate(true_y))

class EpochResult(NamedTuple):
    id: str
    epoch: int
    loss: float
    accuracy: float
    test_accuracy: float

def run_epochs(
    net, criterion, optimizer, train_sets, test_sets, input_size,
    batch_size=200, num_epochs=100, num_minor_epochs=10
):
    train_x, train_y = train_sets
    id = secrets.token_hex(4)

    best_test_accuracy = 0
    counter = 0

    mean, std = train_x.mean(), train_x.std()

    for epoch in trange(num_epochs):
        running_loss, running_accuracy = 0, 0
        shuffled_ids = torch.randperm(train_x.size(0))
        num_batches = train_x.size(0) // batch_size

        for i in range(0, num_batches*batch_size, batch_size):
            optimizer.zero_grad()

            batch_ids = shuffled_ids[i:i+batch_size]
            batch_x = (train_x[batch_ids] - mean) / std
            batch_y = train_y[batch_ids]

            inputs = batch_x.view((batch_size,) + input_size)

            inputs.requires_grad_()

            scores = net(inputs)
            loss = criterion(scores, batch_y)
            loss.backward()

            optimizer.step()

            running_loss += loss.detach().item()

            preds = scores.argmax(dim=1)
            running_accuracy += get_accuracy(batch_y, preds)
    
        loss = running_loss / num_batches
        accuracy = running_accuracy / num_batches

        net.eval()
        test_accuracy, _, _ = eval_test_accuracy(net, mean, std, test_sets, input_size, batch_size=batch_size)
        net.train()

        yield EpochResult(id, epoch, loss, accuracy, test_accuracy)

        # Implement early stopping based on test accuracy

        counter += 1
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            counter = 0
            save_checkpoint(net, id)
        elif counter >= num_minor_epochs:
            break
    
    load_checkpoint(net, id)

def run_experiments(
    init_func, train_sets, test_sets, input_size,
    num_experiments=5, num_epochs=100, num_minor_epochs=10, **kwargs
):
    ids = []
    xs = []
    train_ys = []
    test_ys = []
    runs = []

    for run in range(num_experiments):
        net, criterion, optimizer = init_func()
        for result in run_epochs(
            net, criterion, optimizer, train_sets, test_sets, input_size,
            num_epochs=num_epochs, num_minor_epochs=num_minor_epochs, **kwargs
        ):
            ids.append(result.id)
            xs.append(result.epoch)
            train_ys.append(result.accuracy)
            test_ys.append(result.test_accuracy)
            runs.append(run)

            if result.epoch == 0:
                print(f"Experiment {run} ({result.id}):")
            if result.epoch % num_minor_epochs == 0:
                print(f"epoch = {result.epoch}\t loss = {result.loss:.3f}\t accuracy = {result.accuracy:.3f}\t test accuracy = {result.test_accuracy:.3f}")
        
    return pd.DataFrame(data={
        "Id": ids,
        "Epoch": xs,
        "Train": train_ys,
        "Test": test_ys,
        "Experiment": runs
    })

def plot_experiments(df):
    sns.set_theme(style="whitegrid", font_scale=1.2)

    grid = sns.FacetGrid(
        df.melt(
            id_vars=["Experiment", "Epoch"],
            value_vars=["Train", "Test"],
            var_name="Dataset",
            value_name="Accuracy"
        ),
        col="Dataset", height=6
    )
    num_experiments = df["Experiment"].nunique()
    grid.map_dataframe(sns.lineplot, x="Epoch", y="Accuracy",
        hue="Experiment",
        palette=sns.color_palette("light:#001c75", n_colors=num_experiments))

    sns.set_theme(style="white")

### Evaluation metrics ###

def show_confusion_matrix(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * len(CLASSES), index=CLASSES,
                         columns=CLASSES)

    plt.figure(figsize=(12, 7))
    sns.heatmap(df_cm, annot=True)

def show_eval_metrics(y_pred, y_true, average="macro"):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    print("F1 Score :", f1_score(y_true, y_pred, average="macro"))
    print("Precision:", precision_score(y_true, y_pred, average="macro"))
    print("Recall   :", recall_score(y_true, y_pred, average="macro"))
