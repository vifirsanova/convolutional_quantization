"""
The following is a low-level demonstration version of a compression algorithm proposed by V. Firsanova
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

class DemoDataset():
    def __init__(self):
        """
        Class for building a torch tensor dataset for demonstration
        """
        pass

    def _synthetic_dataset(self):
        """
        Create synthetic dataset to test the algorigthm
        """
        # Set random seed
        np.random.seed(0)
        torch.manual_seed(0)
        # Generate random data points
        X = np.random.rand(1000, 10).astype(np.float32)
        # Generate random target values for binary classification 
        y = (np.sum(X, axis=1) > 5).astype(np.float32) 
        return X, y

    def _load_data(self):
        """
        Load the dataset
        """
        # Generate data points
        X, y = self._synthetic_dataset()
        # Convert numpy arrays to torch tensors
        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y).unsqueeze(1)
        return X_tensor, y_tensor
    
    def _build_dataset(self):
        """
        Create a dataset and split it into training and validation sets
        """
        X_tensor, y_tensor = self._load_data()
        dataset = TensorDataset(X_tensor, y_tensor)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        return train_dataset, val_dataset

    def create_loader(self):
        """
        Create a data loader for the dataset
        """
        train_dataset, val_dataset = self._build_dataset()
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        return train_loader, val_loader

class DemoFFN(nn.Module):
    """
    A simple feed-forward neural network
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the model. Args:
        :param input_dim: input dimension
        :param hidden_dim: hidden dimension
        :param output_dim: output dimension
        """
        super(DemoFFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)   # Input
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Hidden
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # Output

    def forward(self, x):
        """
        Forward pass of the model. Args:
        :param x: input tensor
        :return: output tensor
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class TrainingLoop():
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train_loop(self, train_loader, val_loader, num_epochs=5):
        print('MODEL TRAINING:')
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            # Calculate average training loss
            epoch_loss = running_loss / len(train_loader.dataset)

            # Validate the model
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)

                    # Compute accuracy
                    predicted = (outputs > 0.5).float()
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

            val_loss = val_loss / len(val_loader.dataset)
            val_accuracy = correct / total

            print(f'Epoch {epoch+1}/{num_epochs}, '
                  f'Train Loss: {epoch_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Val Accuracy: {val_accuracy:.4f}')
        print('MODEL TRAINING COMPLETE!\n')

    def get_state_dict(self):
        """
        Get state dictionary for the trained model
        """
        state_dict = self.model.state_dict()
        # Print the state_dict
        print('MODEL WEIGHTS & BIASES SUMMARY:')
        for param_tensor in state_dict:
            print(param_tensor, "\t", state_dict[param_tensor].size())
        return state_dict

class Convolutional_Q():
    """
    Compression algorithm stage 1: Matrix Factorization
    Compression algorithm stage 2: Convolution
    """
    def __init__(self, model, state_dict):
        """
        Initialize the compression algorithm. Args:
        :param model: model to be compressed
        """
        self.model = model
        self.state_dict = state_dict
    
    def matrix_factorization(self, layer, rank):
        """
        Perform matrix factorization on the model weights. Args:
        :layer: weight matrix to be factorized
        :return: reduced weight matrix
        """
        # Get the model weights
        U, S, V = torch.svd(self.model.state_dict()[layer])
        # Choose a rank and reduce
        U_reduced = U[:, :rank]
        S_reduced = S[:rank]
        V_reduced = V[:, :rank]
        # Print the shape
        print('\nLAYER REDUCTION SUMMARY:')
        print('rank \t\t', rank)
        print('U matrix \t', U_reduced.shape)  
        print('S matrix \t', S_reduced.shape)  
        print('V matrix \t', V_reduced.shape)     
        return U_reduced, S_reduced, V_reduced

    def _reconstruct(self, U_reduced, S_reduced, V_reduced, u_conv, s_conv, v_conv):
        """
        Reconstruct the weight matrix
        :return: Reconstructed weight matrix
        """
        # Matrix multiplication to reconstruct weights
        reduced_weight = torch.mm(U_reduced, torch.mm(torch.diag(S_reduced), V_reduced.T))
        # Reconstruct weight matrix
        compressed_weight = u_conv @ s_conv @ v_conv.t()
        print('\nCOMPRESSION SUMMARY:')
        print('Reduced weight \t\t\t', reduced_weight.shape)
        print('Compressed (convolved) weight \t', compressed_weight.shape)
        return reduced_weight, compressed_weight

    def svd_and_convolve(self, layer, rank=4, kernel_size=3):
        """
        Function to apply SVD and convolution. Args:
        :kernel_size: Size of convolutional kernel
        """
        # Perform SVD
        U_reduced, S_reduced, V_reduced = self.matrix_factorization(layer, rank)
        
        # Reshape matrices to fit the convolution operation
        u = U_reduced.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        s = torch.diag(S_reduced).unsqueeze(0).unsqueeze(0)
        v = V_reduced.unsqueeze(0).unsqueeze(0)

        # Define convolution kernel
        kernel = torch.ones((1, 1, kernel_size, kernel_size))  # Simple averaging kernel
        
        # Apply convolution
        u_conv = F.conv2d(u, kernel, padding=kernel_size//2)
        s_conv = F.conv2d(s, kernel, padding=kernel_size//2)
        v_conv = F.conv2d(v, kernel, padding=kernel_size//2)
        
        # Reshape back to original form
        u_conv = u_conv.squeeze(0).squeeze(0)
        s_conv = s_conv.squeeze(0).squeeze(0)
        v_conv = v_conv.squeeze(0).squeeze(0)
        
        # Print the shape
        print('\nLAYER CONVOLUTION SUMMARY:')
        print('kernel size\t', kernel_size)
        print('U matrix \t', u_conv.shape)  
        print('S matrix \t', s_conv[0].shape)  
        print('V matrix \t', v_conv.shape)     

        # Print compression summary
        self._reconstruct(U_reduced, S_reduced, V_reduced, u_conv, s_conv, v_conv)

        return u_conv, s_conv, v_conv

def main():           
    # Build demo dataset
    train_loader, val_loader = DemoDataset().create_loader()

    # Instantiate the demo model
    model = DemoFFN(input_dim=10, hidden_dim=512, output_dim=1)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    training_loop = TrainingLoop(model, criterion, optimizer)
    training_loop.train_loop(num_epochs=5, train_loader=train_loader, val_loader=val_loader)

    # Get state dict
    state_dict = training_loop.get_state_dict()

    # Perform matrix factorization
    u_conv, s_conv, v_conv = Convolutional_Q(model, state_dict).svd_and_convolve(layer='fc2.weight')

    """
    Experiment 1
    """
    # Freeze most of the weights and change top of the matrix for the evaluation
    state_dict_u_conv = state_dict
    state_dict_u_conv['fc2.weight'][:4] = np.transpose(u_conv)
    print('\nUPDATED MATRIX SUMMARY:')
    print('Updated weights:\t', state_dict_u_conv['fc2.weight'][:4].shape, '\t matrix name: u_conv')
    print('Frozen weights:\t\t', state_dict_u_conv['fc2.weight'][4:].shape, '\n')
    # Instantiate the model
    model_u_conv = DemoFFN(input_dim=10, hidden_dim=512, output_dim=1)
    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_u_conv.parameters(), lr=0.001)
    # Load compressed weights
    model_u_conv.load_state_dict(state_dict_u_conv)
    # Train the model to track its progress
    training_loop = TrainingLoop(model_u_conv, criterion, optimizer)
    training_loop.train_loop(num_epochs=2, train_loader=train_loader, val_loader=val_loader)

    """
    Experiment 2
    """
    # Freeze most of the weights and change top of the matrix for the evaluation
    state_dict_v_conv = state_dict
    state_dict_v_conv['fc2.weight'][:4] = np.transpose(v_conv)
    print('UPDATED MATRIX SUMMARY:')
    print('Updated weights:\t', state_dict_v_conv['fc2.weight'][:4].shape, '\t matrix name: v_conv')
    print('Frozen weights:\t\t', state_dict_v_conv['fc2.weight'][4:].shape, '\n')
    # Instantiate the model
    model_v_conv = DemoFFN(input_dim=10, hidden_dim=512, output_dim=1)
    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_v_conv.parameters(), lr=0.001)
    # Load compressed weights
    model_v_conv.load_state_dict(state_dict_v_conv)
    # Train the model to track its progress
    training_loop = TrainingLoop(model_v_conv, criterion, optimizer)
    training_loop.train_loop(num_epochs=2, train_loader=train_loader, val_loader=val_loader)

if __name__=='__main__':
    main()
