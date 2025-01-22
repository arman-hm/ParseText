import os
from tqdm import tqdm
from Levenshtein import distance as levenshtein_distance
import matplotlib.pyplot as plt
import torch



class Train():
    """
    A class to handle the training, validation, checkpointing, and metric evaluation 
    for an OCR model with CTC loss.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function (e.g., CTC Loss).
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to run training on (e.g., 'cuda' or 'cpu').
        char_to_index (dict): Mapping of characters to indices for decoding.
        metrics (list): List of metrics to evaluate during training.
        num_epochs (int): Number of epochs for training.
    """
    def __init__(self, train_loader, val_loader, criterion, optimizer, device,char_to_index ,metrics=list, num_epochs=10):
        """
        Initialize the training class with data loaders, optimizer, device, and metrics.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer =  optimizer
        self.device =  device
        self.char_to_index = char_to_index
        self.metrics = metrics
        self.num_epochs=num_epochs
        self.initial_metrics()

    def initial_metrics(self):
        """
        Initialize metrics tracking for training and validation. 
        Also sets up starting epoch and best metric tracking.
        """
        self.model_results_train = {"Loss": []}
        self.model_results_val = {"Loss":[]}

        for metric in self.metrics:
            self.model_results_val[metric] = []
            self.model_results_train[metric] = []

        self.starting_epoch = 0
        self.best_metrics ={}        

    def fit(self,model,**kwargs):
        """
        Main training loop. Handles training, validation, and checkpointing.

        Args:
            model (torch.nn.Module): The model to be trained.
            **kwargs: Additional parameters like checkpointing and plotting settings.
        """
        self.model = model
        self.model.to(self.device)
        for epoch in range(self.starting_epoch,self.num_epochs):
            self.per_epoch() # Perform training for one epoch
            self.cal_validation() # Perform validation

            # Log training and validation metrics for the epoch
            print("Epoch: ", f"[{epoch+1}/{self.num_epochs}]")
            print(", ".join(f"Train_{key}: {value_list[-1]}" for key, value_list in self.model_results_train.items()))
            print(", ".join(f"Val_{key}: {value_list[-1]}" for key, value_list in self.model_results_val.items()))
            
            # Handle checkpointing
            if 'checkpoint' in kwargs:
                checkpoint = kwargs.get('checkpoint')
                path = checkpoint.get("path") # Default path
                metrics = checkpoint.get("metrics", [])
                log_path = checkpoint.get("log_path")
                self.checkpoint(epoch, path, log_path, metrics)
       
        # Handle optional plotting
        if "plots" in kwargs:
            self.plot(path = kwargs['plots'].get("path"))

    def per_epoch(self):
        """
        Perform one epoch of training. Calculates loss and metrics for training data.
        """
        total_samples = 0
        epoch_results = {"Loss": 0}
        for metric in self.metrics:
            epoch_results[metric] = 0.0
        
        self.model.train()  # Set the model to training mode
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training", leave=False)
        # Iterate on Training Data
        for batch_idx, (images, targets, input_lengths, target_lengths) in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            # Forward pass
            self.model.train()
            logits = self.model(images)
            log_probs = logits.permute(1, 0, 2)

            # Calculate CTC Loss
            loss = self.criterion(log_probs, targets, input_lengths, target_lengths)

            # Backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_results["Loss"] += loss.item()
            
            # Realtime Evaluation each batch
            self.model.eval()
            metrics = self.cal_metrics(log_probs, targets, target_lengths)
            for key, value in metrics.items():
                # Construct the corresponding epoch key
                if key in epoch_results:  # Check if the key exists in epoch_result
                    epoch_results[key] += value

            # set Tqdm output for monitoring
            postfix_dict = {"Loss": epoch_results["Loss"] / (batch_idx + 1)}
            total_samples += len(images)
            for key, value in epoch_results.items():
                if key != "Loss":
                    postfix_dict[key] = value / total_samples

            progress_bar.set_postfix(postfix_dict)

        for key, value in postfix_dict.items():
            self.model_results_train[key].append(value)


    def cal_validation(self):
        """
        Perform one epoch of training. Calculates loss and metrics for training data.
        """
        total_samples = 0
        epoch_val = {"Loss": 0}
        for metric in self.metrics:
            epoch_val[metric] = 0.0
        self.model.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # No gradients needed for validation
            for images, targets, input_lengths,target_lengths in self.val_loader:
                images, targets = images.to(self.device), targets.to(self.device)

                # Forward pass
                logits = self.model(images)
                log_probs = logits.permute(1, 0, 2)

                # Calculate CTC Loss
                loss = self.criterion(log_probs, targets, input_lengths, target_lengths)
                epoch_val["Loss"] += loss.item()
    
                metrics = self.cal_metrics(log_probs, targets, target_lengths)
                for key, value in metrics.items():
                # Construct the corresponding epoch key

                    if key in epoch_val:  # Check if the key exists in epoch_result
                        epoch_val[key] += value
                total_samples += len(images)

            for key, value in epoch_val.items():
                if key == "Loss":
                    self.model_results_val[key].append(value / len(self.val_loader))
                else:
                    self.model_results_val[key].append(value / total_samples)

    def cal_metrics(self, log_probs, targets, target_lengths):
        """
        Calculate metrics like character accuracy, word accuracy, CER, and WER.

        Args:
            log_probs (Tensor): Log probabilities from the model.
            targets (Tensor): Ground truth targets.
            target_lengths (Tensor): Lengths of the targets.

        Returns:
            dict: A dictionary of calculated metrics.
        """

        results = {}
        # Calculate character-level accuracy
        def character_accuracy(predictions, targets):
            correct_chars = 0
            total_chars = 0
            for pred, target in zip(predictions, targets):
                total_chars += len(target)
                correct_chars += sum(p == t for p, t in zip(pred, target))
            return (correct_chars / total_chars if total_chars > 0 else 0.0)*len(targets)

        # Calculate word-level accuracy
        def word_accuracy(predictions, targets):
            correct_words = 0
            for pred, target in zip(predictions, targets):
                if pred == target:
                    correct_words += 1
            return correct_words
        
        # Calculate character-level error rate accuracy
        def cer(predictions, targets):
            character_error_rate = 0
            for pred, target in zip(predictions, targets):
                reference = [list(self.char_to_index.keys())[list(self.char_to_index.values()).index(index)] for index in target]
                hypothesis = [list(self.char_to_index.keys())[list(self.char_to_index.values()).index(index)] for index in pred]
                # Character-level Levenshtein distance
                distance = levenshtein_distance(reference, hypothesis)
                character_error_rate += distance / len(reference) if len(reference) > 0 else float('inf')
            return character_error_rate 
        
        # Calculate word-level error rate accuracy
        def wer(predictions, targets):
            word_error_rate = 0
            total_words = len(targets)
            for pred, target in zip(predictions, targets):
                reference =''.join([list(self.char_to_index.keys())[list(self.char_to_index.values()).index(index)] for index in target ])
                hypothesis = ''.join([list(self.char_to_index.keys())[list(self.char_to_index.values()).index(index)] for index in pred])
                ref_words = reference.split()
                hyp_words = hypothesis.split()
                # Word-level Levenshtein distance
                distance = levenshtein_distance(" ".join(ref_words), " ".join(hyp_words))
                word_error_rate += distance / len(ref_words) if len(ref_words) > 0 else float('inf')
            return word_error_rate
        
        metric_functions = {
            "character_accuracy": character_accuracy,
            "word_accuracy": word_accuracy,
            "WER": wer,
            "CER": cer,
        }
        # extracts the predicted character indices from the log probabilities output
        predictions = log_probs.argmax(2).transpose(0, 1).cpu().numpy()
        
        # Decode predictions and targets to list of character indices
        pred_texts = self.ctc_decode(predictions, blank=0)
        target_texts = [targets[i][:target_lengths[i]].tolist() for i in range(len(targets))]
        
        # Calculate imported metrics
        for metric in self.metrics:
            if metric in metric_functions:
                results[metric] = metric_functions[metric](pred_texts, target_texts)
            else:
                raise ValueError(f"Metric '{metric}' is not implemented.")
        
        return results
  
    def checkpoint(self, epoch, path, log_path, metrics):
        """
        Save the best model checkpoint for each metric in the list when it improves.
        
        Args:
            epoch (int): Current training epoch.
            path (str): Directory to save the model checkpoint.
            metrics (list): List of metrics to monitor (e.g., ['Train_Loss', 'Val_Word_Acc']).
            model_results_train (dict): Dictionary of training metrics.
            model_results_val (dict): Dictionary of validation metrics.
        """
        # Ensure path exists
        os.makedirs(path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        for metric in metrics:
            # Extract the current metric value
            if metric.startswith("Train_"):
                key = metric.replace("Train_", "")
                current_value = self.model_results_train.get(key, [None])[-1]
            elif metric.startswith("Val_"):
                key = metric.replace("Val_", "")
                current_value = self.model_results_val.get(key, [None])[-1]
            else:
                print(f"Metric '{metric}' not available for checkpoint")
                continue

            if current_value is None:
                print(f"Metric '{metric}' not found in the results.")
                continue

            # Check if this is the best value
            is_improved = False
            if metric not in self.best_metrics:
                # Initialize the best metric
                self.best_metrics[metric] = current_value
                is_improved = True
            else:
                # Check if the metric improved (lower for losses, higher for accuracies)
                if ("Loss" in metric or "WER" in metric or "CER" in metric):
                    if current_value < self.best_metrics[metric]:
                        is_improved = True
                else:  # For accuracy metrics
                    if current_value > self.best_metrics[metric]:
                        is_improved = True

            if is_improved:
                print(f"Improved {metric}: {self.best_metrics[metric]} -> {current_value}")
                self.best_metrics[metric] = current_value

                # Save a separate checkpoint for this metric
                checkpoint_path = os.path.join(path, f"best_model_{metric}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_metrics': self.best_metrics,
                    'current_epoch_metrics': {
                        'train': self.model_results_train,
                        'val': self.model_results_val
                    }
                }, checkpoint_path)
                print(f"Model checkpoint saved for metric '{metric}' at '{checkpoint_path}'.")
                                # Save metrics log as .txt
                log = os.path.join(log_path, f"metrics_log_{metric}.txt")
                with open(log, "w") as log_file:
                    log_file.write(f"Checkpoint saved for improved metric: {metric}\n")
                    log_file.write(f"Epoch: {epoch}\n")
                    log_file.write(f"Best {metric}: {current_value}\n")
                    log_file.write(f"\n=== Training Metrics ===\n")
                    for k, v in self.model_results_train.items():
                        log_file.write(f"{k}: {v[-1]}\n")
                    log_file.write(f"\n=== Validation Metrics ===\n")
                    for k, v in self.model_results_val.items():
                        log_file.write(f"{k}: {v[-1]}\n")
                print(f"Metrics log saved at '{log}'.")

    def load_checkpoint(self,model, path:str):
        """
        Load a model checkpoint and restore the state of the model, optimizer, 
        epoch, and metrics.

        Args:
            model (torch.nn.Module): The model to load the checkpoint into.
            path (str): Path to the checkpoint file.
        """
        print(f"Loading checkpoint from: {path}")
        try:
            # Load the checkpoint
            checkpoint = torch.load(path)

            # Restore model and optimizer states
            self.model = model
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.starting_epoch = checkpoint['epoch'] + 1  # Resume from next epoch

            # Restore best metrics
            if 'best_metrics' in checkpoint:
                self.best_metrics = checkpoint['best_metrics']
                print(f"Best metrics restored: {self.best_metrics}")

            # Restore training and validation metrics
            if 'current_epoch_metrics' in checkpoint:
                self.model_results_train = checkpoint['current_epoch_metrics']['train']
                self.model_results_val = checkpoint['current_epoch_metrics']['val']
                print(f"Training and validation metrics restored.")

            print(f"Checkpoint loaded successfully. Resuming from epoch {self.starting_epoch}.")

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
        
        pass
    
    def ctc_decode(self,predictions, blank=0):
        """
        Decode predictions using CTC decoding.

        Args:
            predictions (Tensor): Model predictions.
            blank (int): Blank token index.

        Returns:
            list: Decoded text predictions.
        """
        pred_texts = []
        for pred in predictions:
            pred_chars = []
            previous_char = None
            for char in pred:
                if char != previous_char and char != blank:
                    pred_chars.append(char)
                previous_char = char
            pred_texts.append(pred_chars)
        return pred_texts
    
    def plot(self, path):
        """
        Generate and save plots of training and validation metrics.

        Args:
            path (str): Directory to save the plots.
        """
        for key, value in self.model_results_train.items():
            plt.plot(range(self.num_epochs), value, label=f'Train_{key}')
            plt.plot(self.model_results_val[key], label=f'Val_{key}')
            plt.legend()

            # Save the plot before displaying it
            plt.savefig(os.path.join(path, f"Model_{key}.png"))
            plt.show()  # Call show after saving
            plt.clf() 


    