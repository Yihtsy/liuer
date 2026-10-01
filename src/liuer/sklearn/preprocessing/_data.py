from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class LogisticScaler(BaseEstimator, TransformerMixin):
    """A scaler that transforms data using logistic regression to probability values.
    
    This class uses logistic regression to transform each feature into a probability between 0 and 1.
    """
    
    def __init__(self):
        """Initialize the LogisticScaler."""
        self.model = LogisticRegression(solver='liblinear')
        self.label_encoder = LabelEncoder()  # Initialize LabelEncoder to handle categorical labels
    
    def fit(self, X, y):
        """Fit the logistic regression model on the data.
        
        Parameters:
        X : array-like, shape (n_samples, n_features)
            The input data to fit the model.
        y : array-like, shape (n_samples,)
            The target labels (can be categorical, will be encoded).
        
        Returns:
        self : object
            Fitted transformer.
        """
        # Encode the labels if they are not already numeric (e.g., 'cat' -> 0, 'dog' -> 1)
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Fit the logistic regression model
        self.model.fit(X, y_encoded)
        return self
    
    def transform(self, X):
        """Transform the input data into probabilities using the logistic regression model.
        
        Parameters:
        X : array-like, shape (n_samples, n_features)
            Input data to be transformed.
        
        Returns:
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data (probabilities between 0 and 1).
        """
        # Get the probabilities (output between 0 and 1)
        probabilities = self.model.predict_proba(X)[:, 1]  # Get the probability of class 1
        return probabilities.reshape(-1, 1)  # Return probabilities as a single column

    def fit_transform(self, X, y):
        """Fit the logistic regression model and transform the data."""
        return self.fit(X, y).transform(X)
