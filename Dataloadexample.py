import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset

class DiabetesDataset(Dataset):
      def __init__(self):
          xy = np.loadtxt('./data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
          self.x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
          self.y_data = Variable(torch.from_numpy(xy[:, [-1]]))
          self.len=xy.shape[0]
      def __getitem__(self, index):
          return self.x_data[index],self.y_data[index]

dataset=DiabetesDataset()
train_loader=DataLoader(dataset=dataset,
                        batch_size=32,
                        shuffle=True,
                        num_workers=2) #multi-process
#auto generate batches

class Model(torch.nn.Module):

    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred

# our model
model = Model()
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for epoch in range(2):
    for i,data in enumerate(train_loader,0):
        inputs,labels=data
        inputs,labels=Variable(inputs),Variable(labels)
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(inputs)

        # Compute and print loss
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.data[0])

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()