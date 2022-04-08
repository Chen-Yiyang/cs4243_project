from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

classes = ['terrier', 'bulldog', 'schnauzer', 'sheepdog', 'retriever',
           'spaniel', 'sheepdog', 'setter', 'hound', 'poodle']

def show_confusion_matrix(scores, y_true):
    y_pred = scores.argmax(dim=1).numpy()

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in classes],
                         columns=[i for i in classes])

    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
