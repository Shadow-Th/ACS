import numpy as np

def reinforcement_loop(model, X_train, y_train):
    feedback = []
    preds = model(torch.FloatTensor(X_train)).detach().numpy().flatten()
    for pred, true in zip(preds, y_train):
        reward = 1 if (pred > 0.5) == true else -1
        feedback.append(reward)
    print("Total reward:", sum(feedback))
