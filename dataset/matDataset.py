
from torch.utils.data import Dataset
import torch, scipy.io
import numpy as np
import os


datasetPath = './dataset/BCICIV_2a_mat'


# DA using sliding window
def slidingWindow(data, label):
    data_ = []
    label_ = []
    for i in range(0, 562 - 125, 12):
        data_.append(data[:, :, i:i + 125])
        label_.append(label)
    data = np.concatenate(data_)
    label = np.concatenate(label_)

    return data, label

def readDateset(subjects = ['S01'], dataset = 'train', DA = False): 

    if dataset == 'train':
        # read train dataset
        train_dataset = []
        for subject in subjects:
            filename = os.path.join(datasetPath, 'BCIC_' + subject + '_T.mat')
            train_dataset.append(scipy.io.loadmat(filename))

        x_train = np.concatenate(
            [data['x_train'] for data in train_dataset]
        )
        y_train = np.concatenate(
            [data['y_train'] for data in train_dataset]
        )
        if DA : return slidingWindow(x_train, y_train)
        return x_train, y_train

    elif dataset == 'test':
        # read test dataset
        test_dataset = []
        for subject in subjects:
            filename = os.path.join(datasetPath, 'BCIC_' + subject + '_E.mat')
            test_dataset.append(scipy.io.loadmat(filename))

        x_test = np.concatenate(
            [data['x_test'] for data in test_dataset]
        )
        y_test = np.concatenate(
            [data['y_test'] for data in test_dataset]
        )
        if DA : return slidingWindow(x_test, y_test)
        return x_test, y_test

    elif dataset == 'both':
        
        x_train, y_train = readDateset(subjects, 'train')
        x_test, y_test = readDateset(subjects, 'test')

        data = np.concatenate((x_train, x_test))
        label = np.concatenate((y_train, y_test))

        if DA : return slidingWindow(data, label)
        return data, label

    return



class BCI_Dataset(Dataset):
    def __init__(self, test_subjects = ['S01'], dataset = 'train', DA = False):
        self.data, self.label = readDateset(test_subjects, dataset, DA)
        self.data = torch.tensor(self.data).unsqueeze(1).float()
        self.label = torch.tensor(self.label).squeeze()

    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return len(self.data)


# Dataset for individual subject training scheme
class ID_dataset(BCI_Dataset):
    def __init__(self, test_subject = 'S01', train = True, DA = False):
        if train:
            super().__init__([test_subject], 'train', DA)
        else:
            super().__init__([test_subject], 'test', DA)



# Dataset for subeject independent training scheme
class SI_dataset(BCI_Dataset):
    def __init__(self, test_subject = 'S01', train = True, DA = False):
        subjects = []
        if train:
            for _ in range(1, 10):
                subject = 'S0' + str(_)
                if subject != test_subject :
                    subjects.append(subject)
            super().__init__(subjects, 'both', DA)
        else:
            super().__init__([test_subject], 'test', DA)
    


# Dataset for subject dependent training scheme
class SD_dataset(BCI_Dataset):
    def __init__(self, test_subject = 'S01', train = True, DA = False):
        if train:
            subjects = []
            subjects_without_test = []
            for _ in range(1, 10):
                subject = 'S0' + str(_)
                subjects.append(subject)
                if subject == test_subject: continue
                subjects_without_test.append(subject)
            
            data_all = readDateset(subjects, 'train', DA)
            data_without = readDateset(subjects_without_test, 'test', DA)
            
            self.data = np.concatenate((data_all[0], data_without[0]))
            self.label = np.concatenate((data_all[1], data_without[1]))

            self.data = torch.tensor(self.data).unsqueeze(1).float()
            self.label = torch.tensor(self.label).squeeze()

        else:
            super().__init__([test_subject], 'test', DA)




