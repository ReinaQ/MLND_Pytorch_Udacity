import torch
from torchvision import datasets, transforms

def prepare_data(data_directory_path,is_train):
    ''' Reurn the dataloader of transformed image data from given directory.
        
     Arguments
     ---------
     data_directory_path: string, the actural path of image data(for training, or validation, or test)
     is_train: bool, True for training data, False for validation and test data
     pretrained: bool, whether load the pre-trained parameters for the arch
     '''
    if is_train == True:
        transform = transforms.Compose([transforms.RandomRotation(30),
                              transforms.RandomResizedCrop(224),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        transform = transforms.Compose([transforms.Resize(255),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])


    # Load each dataset with ImageFolder
    data = datasets.ImageFolder(data_directory_path, transform=transform)
    return data

def dataloader(data, is_train):

    # Using the image datasets and the trainforms, define the dataloaders
    dataloader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=is_train)
    
    return dataloader
    
    
 