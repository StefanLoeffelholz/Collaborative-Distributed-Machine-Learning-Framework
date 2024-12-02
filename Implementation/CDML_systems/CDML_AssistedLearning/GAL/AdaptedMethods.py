from ...base import Base_Constants
from ...base import Impl_Methods
import torch



class GradientAssistedLearning():
    def trainMLModelAssisted(self, epochs=None):
        """Train the network."""
        # Define loss and optimizer
        if epochs is None:
            epochs = 1
        residual_store = []
        self.ML_Model.load_state_dict(torch.load("./param_after_central_lr_01", weights_only=True))
        self.model_optimizer = self.optimizer(self.ML_Model.parameters(), lr=0.1, momentum=0.9)
        for epoch in range(epochs):
            self.ML_Model.to(self.device)
            self.ML_Model.train()
            #self.ML_Model.eval()
            epoch_loss = 0.0
            for data, target in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)
                self.model_optimizer.zero_grad()

                output = self.ML_Model(data)
                adjusted_target = []
                for tar in target:
                    tensor = tar - self.pseudo_residuals[tar]
                    tensor = float(tensor.item())
                    tensor = round(tensor)
                    tensor = torch.tensor(tensor, device=self.device)
                    adjusted_target.append(tensor)
                adjusted_target = torch.stack(adjusted_target)
                adjusted_target.to(self.device)
                # Collect gradients from model's parameters
                
                loss = self.criterion(output, adjusted_target) 
                loss.backward()
            
                self.model_optimizer.step()
                epoch_loss += loss.item()
                # Detach and clone to prevent accumulation

                # Save residuals to disk in batches
                  # Explicitly delete variables to free memory
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(self.trainloader)}")
        self.ML_Model.eval()
        for data, target in self.trainloader:
            data_, target_ = data.to(self.device), target.to(self.device)
            # zero the parameter gradients
            # forward + backward + optimize
            output = self.ML_Model(data_)

            # print statistics
            if output is not None:
                residual_store.append(output.detach().clone()) 
            del data_, target_, output

        average_fitted_values = None
        averaged_tensors = [tensor.mean(dim=0) for tensor in residual_store]
        average_fitted_values=torch.stack(averaged_tensors).mean(dim=0)
        self.interimResult = average_fitted_values.detach().clone()

    def updateMLModelAssistedLearning(self, epochs = None):
        residuals = []
        alpha = 0.5
        fitted_values = []
        for res in self.receivedInterimResults: fitted_values.append(res[0])
        
        if fitted_values is not None and fitted_values!=[]:
            fitted_values = torch.stack(fitted_values)
            fitted_values = torch.mean(fitted_values, dim=0)
        if epochs is None:
            epochs = 1
        print("Training on: " + str(len(self.trainloader)) + " batches")
        for epoch in range(epochs):
            self.ML_Model.train()
            self.ML_Model.to(self.device)
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                data, target = data[0], data[1]
                target_copy = target.clone().to(self.device)
                data, target = data.to(self.device), target.to(self.device)
                self.model_optimizer.zero_grad()
                if fitted_values is None or fitted_values==[]:
                    output = self.ML_Model(data)
                else:
                    alice_predictions = self.ML_Model(data)
                    combined_predictions = []

                    for tens in alice_predictions:
                        combined_predictions.append(alpha*tens + fitted_values*(1-alpha))
                    output = torch.stack(combined_predictions, dim=0)
                    output.to(self.device)
                loss = self.criterion(output, target)
                loss.backward()
                self.model_optimizer.step()

                # Collect gradients from model's parameters
                
                self.ML_Model.eval()

                output = self.ML_Model(data).detach().clone()
                if output is not None:
                    residual = []
                    for tar, out in zip(target_copy, output):
                        target_tensor = torch.zeros(out.shape[0]).to(self.device)
                        target_tensor[tar] = 1
                        residual.append(target_tensor - out)
                    residual = torch.stack(residual)
                    residuals.append(residual.detach().clone())  # Detach and clone to prevent accumulation
                
                self.ML_Model.train()

                running_loss += loss.item()

                del data, target, output  # Explicitly delete variables to free memory
                if i % 100 == 99:  # print every 100 mini-batches
                    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                    self.training_loss.append(running_loss / 2000)
                    running_loss = 0.0

        #Calculate the residuals and return them
        averaged_tensor = None
        averaged_tensors = [tensor.mean(dim=0) for tensor in residuals]
        averaged_tensor=torch.stack(averaged_tensors).mean(dim=0)
        for agent in self.agent_ready_to_update:
            if agent == self.id:
                self.pseudo_residuals = averaged_tensor
            else:
                for trainer in self.trainers:
                    if trainer.id == agent:
                        update_message = Impl_Methods.Message(tag=Base_Constants.possible_tags[15], object=averaged_tensor)
                        self.communication_protocoll.send_information(self, message=update_message, communication_address=trainer.ip)
                        break
            
        self.receivedInterimResults = []
    
    def awaitUpdateAssisted(self):
        if self.id != self.updater.id:
            while True:
                update = self.communication_protocoll.receive_information(self)
                if update.tag == Base_Constants.possible_tags[15]:
                    self.pseudo_residuals = update.object
                    break






