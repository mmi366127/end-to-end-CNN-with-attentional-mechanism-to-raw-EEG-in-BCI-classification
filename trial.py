

# Dataset 
from dataset.matDataset import ID_dataset, SI_dataset, SD_dataset
# from dataset.npzDataset import BCI_Dataset
from torch.utils.data import DataLoader

# Learning 
from sherpa.algorithms import Genetic
import torch.optim as optim
import torch.nn as nn
import sherpa, torch

# Models 
from models import *

# Other utils
from tqdm import tqdm
from utils import *
import argparse

# test function
def test(model, device, test_dataset, loss_func):
    # switch to eval
    model.eval()
    loss_total = 0

    with torch.no_grad():
        inputs = test_dataset.data.to(device)
        labels = test_dataset.label.to(device)

        outputs = model(inputs)   
        loss = loss_func(outputs, labels)

        loss_total += loss.item()
        
        correct = (torch.argmax(outputs, dim = 1).squeeze() == labels).sum().item()

    return loss_total / len(test_dataset), correct / len(test_dataset)


# training function
def train(model, device, loss_func, optimizer, max_epoch, train_loader, test_dataset):
    
    loss_log = np.zeros((2, max_epoch))
    accuracy_log = np.zeros((2, max_epoch))
    
    model.to(device)
    for epoch in range(max_epoch):
        batch_count = 0
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        with tqdm(train_loader, unit = 'batch') as tepoch:
            for data, target in tepoch:
                
                batch_count += 1.0
                tepoch.set_description(f'Training Epoch {epoch+1:02d}/{max_epoch:02d}')

                model.train()

                data = data.to(device, dtype = torch.float)
                target = target.to(device)

                optimizer.zero_grad()
                pred = model(data)

                loss = loss_func(pred, target)
                loss.backward()
                optimizer.step()

                correct = (torch.argmax(pred, dim = 1).squeeze() == target).sum().item()
                
                epoch_loss += loss.item() / len(data)
                epoch_accuracy += correct / len(data)

                tepoch.set_postfix(loss = loss.item() / len(data), accuracy = 100 * correct / len(data))

        # epoch_loss /= batch_count
        epoch_accuracy /= batch_count

        current_loss, current_accuracy = test(model, device, test_dataset, loss_func)
        print('current loss: {}, current accuracy: {}'.format(current_loss, current_accuracy))

        loss_log[0][epoch] = epoch_loss
        loss_log[1][epoch] = current_loss

        accuracy_log[0][epoch] = epoch_accuracy
        accuracy_log[1][epoch] = current_accuracy

    return model, loss_log, accuracy_log

        
# Tunning the model with sherpa
def tunning(model_type, train_dataset, test_dataset):
    max_epochs = 150
    # If gpu is available
    if torch.cuda.is_available():  
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    parameters = [
        sherpa.Choice('activation', [nn.ELU, nn.ReLU]),
        sherpa.Continuous('dropout', [0, 0.9]),
        sherpa.Choice('kernel_size', [10, 15, 25]),
        sherpa.Continuous('learning_rate', [0.0001, 0.1]),
        # sherpa.Continuous('learning_rate_decay', [0.5, 1.0]),
        sherpa.Discrete('dense_size', [8, 512]),
        sherpa.Choice('filters', [16, 32, 64]),
        sherpa.Choice('optimizer', [optim.Adam, optim.SGD, optim.RMSprop]),
        sherpa.Discrete('batch_size', [16, 512])
    ]

    algorithm = Genetic(max_num_trials = 100)

    loss_func = nn.CrossEntropyLoss()

    # search for higher accuracy
    study = sherpa.Study(
        parameters = parameters,
        algorithm = algorithm,
        lower_is_better = False,
        disable_dashboard = True
    )
    
    for trial in study:
        print('Trial {}:\t{}'.format(trial.id, trial.parameters))
        train_loader = DataLoader(
            train_dataset, 
            batch_size = trial.parameters['batch_size']
        )
        model = AttentionNet(
            input_size = (22, 125),
            activate_func = trial.parameters['activation'],
            dropout = trial.parameters['dropout'],
            kernel_size = trial.parameters['kernel_size'],
            dense_size = trial.parameters['dense_size'],
            filter_number = trial.parameters['filters']
        )
        optimizer = trial.parameters['optimizer'](
            model.parameters(),
            lr = trial.parameters['learning_rate'],
            weight_decay = 0.1
        )
        model.to(device)
        for epoch in range(max_epochs):
            with tqdm(train_loader, unit = 'batch') as tepoch:
                for inputs, labels in tepoch:
                    tepoch.set_description(f'Training Epoch {epoch+1:03d}/{max_epochs:03d}')
                
                    # set model to train
                    model.train()
                
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                
                    optimizer.zero_grad()

                    outputs = model(inputs)
                    loss = loss_func(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    correct = (torch.argmax(outputs, dim = 1).squeeze() == labels).sum().item()
                    tepoch.set_postfix(loss = loss.item() / len(labels), accuracy = 100 * correct / len(labels))

            current_loss, current_accuracy = test(model, device, test_dataset, loss_func)
            print('current loss: {}, current accuracy: {}'.format(current_loss, current_accuracy))

            study.add_observation(
                trial = trial, 
                iteration = epoch, 
                objective = current_accuracy
            )
    
            if study.should_trial_stop(trial):
                break
        study.finalize(trial = trial)
    
    print(study.get_best_result())


if __name__ == '__main__':
    train_dataset = SI_dataset()
    test_dataset = SI_dataset(train = False)
    tunning('A', train_dataset, test_dataset)

    
"""
full self atten in paper + mul input
{'Trial-ID': 46, 'Iteration': 148, 'activation': <class 'torch.nn.modules.activation.ELU'>, 'batch_size': 83, 'dense_size': 279, 'dropout': 0.491454253367649, 'filters': 32, 'kernel_size': 10, 'learning_rate': 0.013243591021436728, 'optimizer': <class 'torch.optim.adam.Adam'>, 'Objective': 0.4270833333333333}


self atten (no mul w^T)
{'Trial-ID': 79, 'Iteration': 61, 'activation': <class 'torch.nn.modules.activation.ELU'>, 'batch_size': 96, 'dense_size': 50, 'dropout': 0.4998838855278078, 'filters': 16, 'kernel_size': 25, 'learning_rate': 0.050870238623487864, 'optimizer': <class 'torch.optim.sgd.SGD'>, 'Objective': 0.4236111111111111}


full self atten
{'Trial-ID': 81, 'Iteration': 65, 'activation': <class 'torch.nn.modules.activation.ELU'>, 'batch_size': 58, 'dense_size': 176, 'dropout': 0.2952126353064908, 'filters': 32, 'kernel_size': 10, 'learning_rate': 0.08357707054864591, 'optimizer': <class 'torch.optim.sgd.SGD'>, 'Objective': 0.4583333333333333}



"""