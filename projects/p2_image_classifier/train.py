import argparse
import torch
from torch import nn
from torch import optim
import network
from utility import prepare_data, dataloader

def train(model, lr_rate, trainloader, validationloader,epochs, use_gpu):
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    steps = 0
    running_loss = 0
    print_every = 50
    # Define the loss
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    if model.arch == 'resnet152':
        optimizer = optim.Adam(model.model.fc.parameters(), lr=lr_rate)
    elif model.arch == 'densenet169':
        optimizer = optim.Adam(model.model.classifier.parameters(), lr=lr_rate)
        
    for e in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            
            # Move input and label tensors to the default device(GPU)
            inputs, labels = inputs.to(device), labels.to(device)

            # Clear all the gradients
            optimizer.zero_grad()
            
            # Move model to the default device(GPU)
            model.to_device(device)

            # Forward-feeding, get the loss with criterion, backpropagate, update the weights with optimizer
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Validation process
            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                
                # Model in prediction mode, turn off dropout 
                model.model.eval()
                
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    for inputs, labels in validationloader:
                        ## Move input and label tensors to the default device(GPU)
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        logps = model.forward(inputs)
                        validation_loss += criterion(logps, labels).item()

                        # Calculate accuracy
                        # Model's output is log-softmax, take exponential to get the probabilities
                        ps = torch.exp(logps)
                        # Class with highest probability is our predicted class, compare with true label
                        top_p, top_class = ps.topk(1, dim=1)
                        equality = top_class == labels.view(*top_class.shape)
                        # Accuracy is number of correct predictions divided by all predictions, just take the mean
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                print(f'Epoch {e+1}/{epochs}.. '
                      f'Train loss: {running_loss/print_every:.5f}.. '
                      f'Validation loss: {validation_loss/len(validationloader):.5f}.. '
                      f'Valication accuracy: {accuracy/len(validationloader):.5f}')
                running_loss = 0
                # Make sure dropout and grads are on for training
                model.model.train()
    return model

def save_checkpoints(data, model, checkpoint_path):
    ''' Reurn the checkpoint information of trained model.
        
     Arguments
     ---------
     data: ImageFolder(for training datatset, or validation dataset, or test dataset)
     '''
    checkpoint = {'class_to_idx': data.class_to_idx,
              'state_dict': model.state_dict(),
              'arch': model.arch,
              'hidden_units': model.hidden_units}

    torch.save(checkpoint, checkpoint_path)
    
    
parser = argparse.ArgumentParser(description='Image Classifier.')

parser.add_argument('--epochs', type=int,help='epoch of model training')
parser.add_argument('datadirectory', type=str,
              help='the path for training data, or validation data, or test data')
parser.add_argument('--learning_rate', type=float,
              help='learning rate for gradient descent')
parser.add_argument('--gpu', action='store_true', help='use gpu or not')
parser.add_argument('--arch', choices=['resnet152', 'densenet169'], default='resnet152', help='the name of pre-trained model')
parser.add_argument('--save_dir', type=str, help='the name of checkpoint file')
parser.add_argument('--hidden_units', type=int, help='the number of hidden units')

args = parser.parse_args()

data_dir = args.datadirectory
save_dir = args.save_dir
arch = args.arch 
hidden_units = args.hidden_units
epochs = args.epochs if args.epochs else 10
lr = args.learning_rate if args.learning_rate else 0.001
use_gpu = True if args.gpu else False

if hidden_units:
    model = network.Model(arch, hidden_units, pretrained=True)
else:
    model = network.Model(arch, pretrained=True)

train_data = prepare_data(data_dir + '/train/', True)
validation_data = prepare_data(data_dir + '/valid/', False)
train_dataloader = dataloader(train_data, True)
validation_dataloader = dataloader(validation_data, False)
trained_model = train(model, lr, train_dataloader, validation_dataloader, epochs, use_gpu)

if save_dir:
    save_checkpoints(train_data, trained_model, save_dir)

else:
    print('Training is finished, no directory found to save model')

