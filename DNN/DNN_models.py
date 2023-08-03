from torch import nn



# Origninal simple DNN model
class DNN_simple(nn.Module):
    def __init__(self, input_vars):
        super(DNN_simple, self).__init__()
        self.layer1 = nn.Linear(len(input_vars), 2)
        # self.layer2 = nn.Linear(3, 2)
        # self.layer3 = nn.Linear(128, 64)
        # self.layer4 = nn.Linear(64, 32)
        self.layer5 = nn.Linear(2, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        # x = self.relu(self.layer2(x))
        # x = self.relu(self.layer3(x))
        # x = self.relu(self.layer4(x))
        x = self.sigmoid(self.layer5(x))
        return x

class DNN_bestFeature(nn.Module):
    def __init__(self, input_vars):
        super(DNN_bestFeature, self).__init__()
        self.layer1 = nn.Linear(len(input_vars), 40)
        self.layer2 = nn.Linear(40, 30)
        self.layer3 = nn.Linear(30, 20)
        self.layer4 = nn.Linear(20, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.layer4(x))
        return x
    

class DNN_flexible(nn.Module):
    def __init__(self, input_vars, hidden_layer_sizes):
        super(DNN_flexible, self).__init__()

        layer_sizes = [len(input_vars)-2] + hidden_layer_sizes + [1]

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
        x = self.sigmoid(self.layers[-1](x))
        return x
