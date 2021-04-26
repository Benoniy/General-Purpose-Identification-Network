

class Model:
    def __init__(self, name, size, epochs, batch_size, data_path):
        self.NAME = name
        self.SIZE = size
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size
        self.data_path = data_path

    def __bool__(self):
        return True
