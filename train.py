import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def train(model, dataset, epochs=2):
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # Dummy training loop
    accs = [0.4, 0.55]
    plt.plot(range(1, len(accs)+1), accs)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("results/accuracy.png")

    cm = np.array([[5,2],[1,7]])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("results/confusion_matrix.png")
    print("Training complete, results saved.")

if __name__ == "__main__":
    print("Dummy training run - replace with actual code")