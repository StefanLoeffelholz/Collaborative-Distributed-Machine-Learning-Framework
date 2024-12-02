from ...base import Base_Constants
from ...base import Impl_Methods
from ...base import Base_Model
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision

class SplitLearningVanilla():
    def trainMLModel(self, epochs=None):
        """Train the network."""
        # Define loss and optimizer
        if epochs is None:
            epochs = 1
        print(f"Training {epochs} epoch(s) w/ {len(self.trainloader)} batches each")
        #print(self.MLTask)
        interimResult = []
        # Train the network
        self.ML_Model.train
        self.ML_Model.to(self.device)
        for epoch in range(epochs):  # loop over the dataset multiple times
            for i, data in enumerate(self.trainloader, 0):
                images, labels = data[0].to(self.device), data[1].to(self.device)
                self.model_optimizer.zero_grad()
                outputs = self.ML_Model(images)
                outputs = outputs.detach().clone()
                labels = labels.detach().clone()
                outputs.requires_grad=True
                interimResult.append((outputs, labels))
        
        if(self.id == self.updater.id): 
            for output_label in interimResult:
                outputs = output_label[0]
                labels = output_label[1]
                self.ML_Model_update.train
                self.ML_Model_update.to(self.device)

                optimizer = self.model_optimizer
                optimizer_2 = self.update_optimizer
                optimizer_2.zero_grad()
                
                data, labels = outputs.to(self.device), labels.to(self.device)
                data.requires_grad = True
                server_output = self.ML_Model_update(data)
                loss = self.criterion(server_output, labels)

                
                loss.backward(retain_graph=True)
                optimizer_2.step()

                data_grad = outputs.grad
                client_grad = data_grad.detach().clone()

                outputs.backward(client_grad)
                optimizer.step()
        
        self.interimResult = (interimResult, self.ip)
        try:
            self.testMLModel()
        except:
            pass
    
    def updateMLModelSplitLearning(self):
        torch.autograd.set_detect_anomaly(True)
        self.ML_Model_update.to(self.device)
        optimizer = self.update_optimizer
        server_grads = []
        for result in self.receivedInterimResults:
            store = []
            ip = result[1]
            if ip != self.ip:
                for data_label in result[0]:
                    optimizer.zero_grad()
                    data, label = data_label[0].to(self.device), data_label[1].to(self.device)

                    self.ML_Model_update.train()
                    data.requires_grad = True

                    server_output = self.ML_Model_update(data)
                    loss = self.criterion(server_output, label)

                    loss.backward(retain_graph=True)
                    optimizer.step()
                    data_grad = data.grad
                    client_grad = data_grad.detach().clone()
                    store.append(client_grad)
                server_grads.append((store, ip))
        
        for data in server_grads:
            update_message = Impl_Methods.Message(tag=Base_Constants.possible_tags[15], object=data[0])
            self.communication_protocoll.send_information(self, message=update_message, communication_address=data[1])
            
        self.receivedInterimResults = []
        print(loss.item)
    
    def awaitUpdate(self):
        if self.id != self.updater.id:
            while True:
                update = self.communication_protocoll.receive_information(self)
                if update.tag == Base_Constants.possible_tags[15]:
                    self.ML_Model.train
                    optimizer = self.model_optimizer
                    for loss, grad in zip(self.interimResult[0], update.object):
                        loss = loss[0]
                        loss.backward(grad)     
                        # Update client-side model parameters
                        optimizer.step()
                    # Update client-side model parameters
                    break

    def testMLModel(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        criterion = nn.CrossEntropyLoss()
        testset = torchvision.datasets.CIFAR100(root='./data_cifar100', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
        model_client = self.ML_Model
        model_server = self.ML_Model_update
        combined_model = Base_Model.Combined_Model(model_server, model_client)
        combined_model.to(self.device)
        combined_model.eval
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = combined_model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Test Loss: {test_loss/len(testloader)}')
        print(f'Accuracy: {100 * correct / total}%')
        return f'Test Loss: {test_loss/len(testloader)}', f'Accuracy: {100 * correct / total}%'






