from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

classes = ['terrier', 'bulldog', 'schnauzer', 'sheepdog', 'retriever',
           'spaniel', 'sheepdog', 'setter', 'hound', 'poodle']

def show_confusion_matrix(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in classes],
                         columns=[i for i in classes])

    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)

def show_eval_metrics(y_pred, y_true, average="macro"):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    print("F1 Score :", f1_score(y_true, y_pred, average="macro"))
    print("Precision:", precision_score(y_true, y_pred, average="macro"))
    print("Recall   :", recall_score(y_true, y_pred, average="macro"))


