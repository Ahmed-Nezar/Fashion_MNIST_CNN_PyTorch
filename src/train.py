from load_data import *
from model import *

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
validation_loader = DataLoader(dataset=validation_data, batch_size=64, shuffle=False)

input_channels = 1
output_channels = 100

model = ConvNet(input_channels, output_channels)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

cost_list=[]
accuracy_list=[]
N_test=len(validation_data)
COST=0

def train_model(n_epochs):
    for epoch in range(n_epochs):
        COST=0
        COST_VAL=0
        for x, y in train_loader:
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            COST+=loss.data
        
        cost_list.append(COST)
        correct=0
        #perform a prediction on the validation  data  
        for x_test, y_test in validation_loader:
            z = model(x_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
            validation_loss = criterion(z, y_test)
            COST_VAL+=validation_loss.data
        accuracy = correct / N_test
        accuracy_list.append(accuracy)
        print("Epoch: {} - Loss: {} - Validation Loss:{} - Accuracy: {}".format(epoch+1, COST,COST_VAL,accuracy))

train_model(6)