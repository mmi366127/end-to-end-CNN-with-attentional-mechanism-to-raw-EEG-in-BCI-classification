
from torch.utils.data import Dataset
import numpy as np
import torch
import os

datasetPath = './dataset/BCICIV_2a'

# DA using sliding window, sample rate = 250 Hz, 750 samples(3s)
def slidingWindow(data, label):
    data_ = []
    label_ = []
    for i in range(0, 500, 25):
        data_.append(data[:, :, i:i + 250])
        label_.append(label)
    data = np.concatenate(data_)
    label = np.concatenate(label_)

    return data, label

def readDataset(subjects = ['01'], dataset = 'train', DA = False):

    # Types of motor imagery
    #           left    roght   foot    tongue
    MI_types = {769: 0, 770: 1, 771: 2, 772: 3}

    if dataset == 'train':
        
        x_train, y_train = [], []
        for subject in subjects:
            filename = os.path.join(datasetPath, 'A' + subject + 'T.npz')
            data = np.load(filename)
            
            startrial_code = 768
            
            raw = data['s'].T
            events_type = data['etyp'].T
            events_position = data['epos'].T

            startrial_events = events_type == startrial_code
            indices = [i for i, x in enumerate(startrial_events[0]) if x]

            trials = []
            classes = []
            for index in indices:
                try:
                    type_e = events_type[0, index+1]
                    class_e = MI_types[type_e]
                    classes.append(class_e)

                    # crop the motor imagery part
                    start = events_position[0, index]
                    trial = raw[: 22, start + 750: start + 1500] 
                    trials.append(trial)

                except:
                    continue

            x_train.append(np.array(trials))
            y_train.append(np.array(classes))

        x_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train)

        if DA : return slidingWindow(x_train, y_train)
        return x_train, y_train
            
    elif dataset == 'test':
        
        x_test, y_test = [], []
        for subject in subjects:
            filename = os.path.join(datasetPath, 'A' + subject + 'E.npz')
            data = np.load(filename)
            
            startrial_code = 768
            
            raw = data['s'].T
            events_type = data['etyp'].T
            events_position = data['epos'].T

            startrial_events = events_type == startrial_code
            indices = [i for i, x in enumerate(startrial_events[0]) if x]

            trials = []
            classes = []
            for index in indices:
                try:
                    type_e = events_type[0, index+1]
                    class_e = MI_types[type_e]
                    classes.append(class_e)

                    # crop the motor imagery part
                    start = events_position[0, index]
                    trial = raw[: 22, start + 750: start + 1500] 
                    trials.append(trial)

                except:
                    continue

            x_test.append(np.array(trials))
            y_test.append(np.array(classes))

        x_test = np.concatenate(x_test)
        x_test = np.concatenate(y_test)

        if DA : return slidingWindow(x_test, y_test)
        return x_test, y_test

    elif dataset == 'both':

        x_train, y_train = readDataset(subjects, 'train')
        x_test, y_test = readDataset(subjects, 'test')

        data = np.concatenate((x_train, x_test))
        label = np.concatenate((y_train, y_test))

        if DA : return slidingWindow(data, label)
        return data, label

class BCI_Dataset(Dataset):
    def __init__(self, test_subjects = ['01'], dataset = 'train', DA = False):
        self.data, self.label = readDataset(test_subjects, dataset, DA)
        self.data = torch.tensor(self.data).unsqueeze(1).float()
        self.label = torch.tensor(self.label).squeeze()

    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return len(self.data)


# Dataset for individual subject training scheme
class ID_dataset(BCI_Dataset):
    def __init__(self, test_subject = '01', train = True, DA = False):
        if train:
            super().__init__([test_subject], 'train', DA)
        else:
            super().__init__([test_subject], 'test', DA)



# Dataset for subeject independent training scheme
class SI_dataset(BCI_Dataset):
    def __init__(self, test_subject = '01', train = True, DA = False):
        subjects = []
        if train:
            for _ in range(1, 10):
                subject = '0' + str(_)
                if subject != test_subject :
                    subjects.append(subject)
            super().__init__(subjects, 'both', DA)
        else:
            super().__init__([test_subject], 'test', DA)



# Dataset for subject dependent training scheme
class SD_dataset(BCI_Dataset):
    def __init__(self, test_subject = '01', train = True, DA = False):
        if train:
            subjects = []
            subjects_without_test = []
            for _ in range(1, 10):
                subject = '0' + str(_)
                subjects.append(subject)
                if subject == test_subject: continue
                subjects_without_test.append(subject)
            
            data_all = readDataset(subjects, 'train', DA)
            data_without = readDataset(subjects_without_test, 'test', DA)
            
            self.data = np.concatenate((data_all[0], data_without[0]))
            self.label = np.concatenate((data_all[1], data_without[1]))

            self.data = torch.tensor(self.data).unsqueeze(1).float()
            self.label = torch.tensor(self.label).squeeze()

        else:
            super().__init__([test_subject], 'test', DA)




