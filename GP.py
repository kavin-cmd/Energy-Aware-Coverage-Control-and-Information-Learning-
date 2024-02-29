import GPy
import numpy as np

class GP:
    def __init__(self, input_dim=2, l=2, sigma_f=0.5):
        self.kernel = GPy.kern.RBF(input_dim=input_dim)
        self.mean = GPy.mappings.Linear(input_dim=input_dim,output_dim=1)
        self.X_train = None
        self.Y_train = None
        self.model = None
    
    def train(self, X_train, Y_train):
        """
        Train the GP model with the provided training data.
        Parameters:
        - X_train: Training input data with shape (num_samples, input_dim).
        - Y_train: Training output data with shape (num_samples, output_dim).
        """
        if self.X_train is None:
            self.X_train = X_train
            self.Y_train = Y_train
        else:
            self.X_train = np.concatenate((self.X_train, X_train), axis=0)
            self.Y_train = np.concatenate((self.Y_train, Y_train), axis=0)   
        
        # Create a GP regression model
        self.model = GPy.models.GPRegression(self.X_train, self.Y_train)
        
        # Optimize the model parameters
        self.model.optimize()  # Turn off messages for brevity
        

    def predict(self, X_test):
        """
        Make predictions with the trained GP model.
        Parameters:
        - X_test: Test input data with shape (num_test_samples, input_dim).
        Returns:
        - pred_mean: Predicted mean with shape (num_test_samples, output_dim).
        - pred_var: Predicted variance with shape (num_test_samples, output_dim).
        """
        if self.X_train is None or self.Y_train is None:
            raise ValueError("Training data is not provided. Use train method to set the training data.")

        if self.model is None:
            raise ValueError("Model not trained. Use train method to train the model.")

        # Make predictions
        pred_mean, pred_var = self.model.predict(X_test)
        return pred_mean, pred_var
    
