from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch

from pathlib import Path

DATA_PATH = Path("data/")

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

def load_tensor(type, size):
    tensor_x = torch.load(DATA_PATH / f"{type}_data_{size}.pt")
    tensor_y = torch.load(DATA_PATH / f"{type}_label_{size}.pt")
    assert tensor_x.is_contiguous()
    assert tensor_y.is_contiguous()
    return (tensor_x, tensor_y)

def save_tensor(tensor_x, tensor_y, type, size):
    assert tensor_x.is_contiguous()
    assert tensor_y.is_contiguous()
    torch.save(tensor_x, DATA_PATH / f"{type}_data_{size}.pt")
    torch.save(tensor_y, DATA_PATH / f"{type}_label_{size}.pt")

### Lab utility functions ###

def display_num_param(net):
    nb_param = 0
    for param in net.parameters():
        nb_param += param.numel()
    print('There are {} ({:.2f} million) parameters in this neural network'.format(
        nb_param, nb_param/1e6)
         )

def show(X):
    if X.dim() == 3 and X.size(0) == 3:
        plt.imshow( np.transpose(  X.numpy() , (1, 2, 0))  )
        plt.show()
    elif X.dim() == 2:
        plt.imshow(   X.numpy() , cmap='gray'  )
        plt.show()
    else:
        print('WRONG TENSOR SIZE')

def get_error( scores , labels ):
    bs=scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches=indicator.sum()
    
    return 1-num_matches.float()/bs  

### Evaluation metrics ###

CLASSES = ['terrier', 'bulldog', 'schnauzer', 'sheepdog', 'retriever',
           'spaniel', 'sheepdog', 'setter', 'hound', 'poodle']

def show_confusion_matrix(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=CLASSES,
                         columns=CLASSES)

    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)

def show_eval_metrics(y_pred, y_true, average="macro"):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    print("F1 Score :", f1_score(y_true, y_pred, average="macro"))
    print("Precision:", precision_score(y_true, y_pred, average="macro"))
    print("Recall   :", recall_score(y_true, y_pred, average="macro"))
