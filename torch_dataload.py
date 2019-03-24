import torch
from torch.utils.data.dataset import Dataset

class MyDataset(Dataset):
    """
    This dataset contains a list of numbers in the range [a,b] inclusive
    """
    def __init__(self, x_train, x_test, y_train, y_test, train = True):
        super(MyDataset, self).__init__()
        self.x_train = torch.tensor(x_train, dtype = torch.float)
        self.x_test = torch.tensor( x_test, dtype = torch.float)
        self.y_train = torch.tensor(y_train, dtype = torch.float)
        self.y_test = torch.tensor(y_test, dtype = torch.float)
        self.train = train
        
    def __len__(self):
        if self.train == True:
            return self.x_train.shape[0]
        else:
            return self.x_test.shape[0]
        
    def __getitem__(self, index):
        if self.train == True:
            return self.x_train[index], self.y_train[index]
        else:
            return self.x_test[index], self.y_test[index]