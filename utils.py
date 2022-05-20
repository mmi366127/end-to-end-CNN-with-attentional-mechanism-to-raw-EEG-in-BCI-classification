from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import torch, scipy.io
import numpy as np

# object for saving training history
class history():
    def __init__(self):
        pass


def validation(model, device, valid_loader, loss_func):

    # switch to eval
    model.eval()
    loss_total = 0

    with torch.no_grad():
        for data in valid_loader:
            inputs = data[0].to(device)
            labels = data[1].to(device)

            outputs = model()   
            loss = loss_func(outputs, labels)

            loss_total += loss.item()

    return loss_total / len(valid_loader)



def train(device, model, epochs, optimizer, loss_func, train_loader, valid_loader, batch_size):
    # set to inf 
    patience = 3
    trigger_times = 0
    last_loss = 1000000000
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        with tqdm(train_loader, unit = 'batch') as tepoch:
            for inputs, labels in tepoch:
                
                tepoch.set_description(f'Training Epoch {epoch+1:03d}/{epochs:03d}')

                # set model to train
                model.train()
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                correct = (torch.argmax(outputs, dim = 1).squeeze() == labels).sum().item()
                tepoch.set_postfix(loss = loss.item() / batch_size, accuracy = 100 * correct / batch_size)


        current_loss = validation(model, device, valid_loader, loss_func)
        print('current loss: {}'.format(current_loss))

        if current_loss > last_loss:
            trigger_times += 1
            
            if trigger_times >= patience:
                print('Early stopping on epoch: {}'.format(epoch))

        last_loss = current_loss

    return model


