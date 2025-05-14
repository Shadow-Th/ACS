from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test)).numpy().flatten()
        preds = (preds > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        print("Accuracy:", acc)
        plt.bar(['Accuracy'], [acc])
        plt.title("Model Accuracy")
        plt.savefig("static/plots/accuracy_plot.png")
        plt.close()