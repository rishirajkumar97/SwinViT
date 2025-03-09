import logging
logger = logging.getLogger(__name__)
class ModelValidator:
    def __init__(self, model, train_loader, val_loader, device, w_train=0.5, w_val=0.5):
        """
        Initialize the ModelValidator.

        Args:
            model: The PyTorch model to validate.
            train_loader: DataLoader for the training dataset.
            val_loader: DataLoader for the validation dataset.
            device: Device to run the model on ('cuda' or 'cpu').
            w_train (float): Weight for training accuracy in the fitness score.
            w_val (float): Weight for validation accuracy in the fitness score.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.w_train = w_train
        self.w_val = w_val
        logging.basicConfig(filename='ga_validator.log', level=logging.INFO)

    def _compute_accuracy(self, weights, data_loader):
        """
        Compute accuracy for a given dataset.

        Args:
            weights: Tensor of weights to set in the last layer.
            data_loader: DataLoader for the dataset to evaluate.

        Returns:
            Accuracy (in percentage) for the dataset.
        """
        # Update the model's weights
        self.model.head.weight.data = weights

        # Set the model to evaluation mode
        self.model.eval()

        correct = 0
        total = 0

        # Disable gradient computation for validation
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        return 100.0 * correct / total

    def validate_model_with_weights(self, weights):
        """
        Validate the model on both training and validation datasets and compute the aggregated fitness score.

        Args:
            weights: Tensor of weights to set in the last layer.

        Returns:
            Aggregated fitness score based on training and validation accuracies.
        """
        # Compute training accuracy
        train_accuracy = self._compute_accuracy(weights, self.train_loader)

        # Compute validation accuracy
        val_accuracy = self._compute_accuracy(weights, self.val_loader)

        # Compute the aggregated fitness score
        fitness_score = self.w_train * train_accuracy + self.w_val * val_accuracy

        print(f"Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%, Fitness Score: {fitness_score:.2f}")

        return fitness_score