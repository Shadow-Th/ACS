import torch
from torch.utils.data import TensorDataset, DataLoader
from model.neural_net import EmailClassifier

model = EmailClassifier()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(X_train, y_train):
    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train.reshape(-1, 1)))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(10):
        for X_batch, y_batch in loader:
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model