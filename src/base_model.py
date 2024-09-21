# base_model.py
class BaseModel:
    def __init__(self, input_dim, output_dim, name="BaseModel"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name

    def train(self, dataset, epochs, batch_size):
        raise NotImplementedError("Subclasses should implement this method.")

    def generate_data(self, length):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def load(self, path):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def save(self, path):
        raise NotImplementedError("Subclasses should implement this method.") 
