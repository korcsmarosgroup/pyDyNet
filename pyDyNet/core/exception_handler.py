class UnImplementedFileFormat(Exception):
    def __int__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"Incorrect File Format Error: {self.message}"
        else:
            return "Incorrect File Format Error."


class IncompatibleDataType(Exception):
    def __int__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"Incorrect Data Type: {self.message}"
        else:
            return "Incorrect Data Type. Please use np.array or pd.DataFrames."


class UnknownWeightType(Exception):
    def __int__(self, weight_type):
        if weight_type:
            self.weight_type = weight_type
        else:
            self.weight_type = None

    def __str__(self):
        if self.weight_type:
            return f"Network weight type {self.weight_type} cannot be extracted."


class NetworkNotInitialized(Exception):
    def __init__(self):
        self.message = "Network not initialized. Please create networks."

    def __str__(self):
        return self.message


class GraphTypeIncompatible(Exception):
    def __init__(self):
        self.message = "PyDyNet can only work with directed or undirected graphs."

    def __str__(self):
        return self.message