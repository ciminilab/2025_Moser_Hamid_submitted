import sys
import os
import torch
from torch.cuda.amp import GradScaler
from neptune.utils import stringify_unsupported

from .vae_model.utils.plot import visualize_model_reconstruction
from .vae_model.linear_vae import VAE_LINEAR
from .vae_model.double_encoder_linear_vae import DOUBLE_ENC_VAE_LINEAR
from .vae_model.cnn_vae import VAE_CNN
from .vae_model.double_decoder_cnn_vae import DOUBLE_DECODER_VAE_CNN
from .vae_model.model_config import CNNVAEConfig, DoubleEncoderLinearVAEConfig, LinearVAEConfig, DoubleDecoderCNNConfig
from .loss_functions.validation_logger import BaseValidationLogger

class Model():
    def __init__(self, model_config_object, cuda_device_num=1, neptune_run=None, validation_logger=None, temp_folder=None, model_tag="", tensor_type=torch.float32) -> None:
        self.hyperparameters = model_config_object.hyperparameters
        self.apply_sigmoid = model_config_object.model_config.apply_sigmoid
        self.device = torch.device(f'cuda:{cuda_device_num}' if torch.cuda.is_available() else 'cpu')
        # If no validation logger is provided, use BaseValidationLogger which does not log anything
        self.validation_logger = validation_logger if validation_logger else BaseValidationLogger()
        temp_folder = temp_folder if temp_folder else "__saved_models"
        if not os.path.exists(temp_folder):
            os.mkdir(temp_folder)
        self.saved_model_path = os.path.join(temp_folder, f"{model_tag}vae_best_epoch.pt")
        self.tensor_type = tensor_type

        self.neptune_run = neptune_run
        if neptune_run:
            neptune_run["model_config"] = stringify_unsupported(vars(model_config_object.model_config))
            neptune_run["hyperparameters"] = stringify_unsupported(vars(model_config_object.hyperparameters))
            neptune_run["hyperparameters/loss_function"] = model_config_object.hyperparameters.loss_function.__name__

        if isinstance(model_config_object, CNNVAEConfig):
            self.vae_model = VAE_CNN(model_config_object.model_config)
        elif isinstance(model_config_object, LinearVAEConfig):
            self.vae_model = VAE_LINEAR(model_config_object.model_config)
        elif isinstance(model_config_object, DoubleEncoderLinearVAEConfig):
            self.vae_model = DOUBLE_ENC_VAE_LINEAR(model_config_object.model_config)
        elif isinstance(model_config_object, DoubleDecoderCNNConfig):
            self.vae_model = DOUBLE_DECODER_VAE_CNN(model_config_object.model_config)


    def test_shape(self, train_loader):
        test_data_iterator = iter(train_loader)
        test_batch = next(test_data_iterator)
        example_images = test_batch[0]
        self.vae_model.test_shape(example_images)

    def _get_best_epoch_model(self):
        vae_best_epoch = self.vae_model
        vae_best_epoch.load_state_dict(torch.load(self.saved_model_path))
        vae_best_epoch.to(self.device)
        vae_best_epoch.eval()
        return vae_best_epoch
    
    def _get_class_weights(self, original_weights):
        """
        Calculate the class weights for the current batch.
        Each class weight is averaged across all samples of the batch and then normalized to the range (1, 10).
        """
        mean_class_weights = torch.mean(original_weights, dim=0)
        if torch.all(torch.eq(mean_class_weights, 1)).item():
            return original_weights

        lower_weight_bound, upper_weight_bound = 0, 1
        min_weight, max_weight = 0, max(mean_class_weights)
        normalized_weights = [x if x==0 else (lower_weight_bound + ((x - min_weight) * (upper_weight_bound - lower_weight_bound)) / (max_weight - min_weight)) for x in mean_class_weights]
        return torch.tensor(normalized_weights)

    def _validate_model_output(self, x_recontsructed):
        """
        Validate the model reconstruction output:  
        Due to numercial instability (usually due to exploading gradients/weights) the model might "collaps" and output NaN values.
        Make sure the model output does not contain any NaN values.  
        If Nan values are found the training run wil be aborted.
        """
        # For each decoder output check if if model output is valid (no NaN values)
        for output in x_recontsructed:
            if torch.any(torch.isnan(output)):
                if self.neptune_run:
                    self.neptune_run["ERROR"] = "NaN values in model output"
                    self.neptune_run["sys/failed"] = True
                    self.neptune_run.stop()
                print("Nan values in model output")
                sys.exit()
 

    def train(self, train_loader, validation_loader):
        # By default (beta = -1) we will use the standard literature weight for the kld loss term
        kl_weight = self.hyperparameters.batch_size / len(train_loader.dataset)
        beta = kl_weight if self.hyperparameters.beta == -1 else self.hyperparameters.beta
        # Specify if the model output are logits or sigmoids
        logit_output = not self.apply_sigmoid
        self.loss_function = self.hyperparameters.loss_function(reduction='mean', beta=beta, logits=logit_output)

        vae_model = self.vae_model.to(self.device)
        self.optimizer = torch.optim.Adam(vae_model.parameters(), lr=self.hyperparameters.learning_rate, eps=self.hyperparameters.epsilon, weight_decay=self.hyperparameters.weight_decay)

        float32_info = torch.finfo(torch.float32)
        min_training_loss = (-1, float32_info.max)
        for epoch in range(self.hyperparameters.epochs):
            self._fit(train_loader)

            epoch_validation_loss = self._validate(validation_loader)

            if epoch_validation_loss < min_training_loss[1]:
                min_training_loss = (epoch, epoch_validation_loss)
                torch.save(vae_model.state_dict(), self.saved_model_path)

            if (epoch+1) % 5 == 0:
                print(f'Epoch: {epoch+1}, validation loss: {epoch_validation_loss:.4f}')

        print(f'Best validation loss: {min_training_loss[1]:.4f} in epoch: {min_training_loss[0]+1}')


    def _fit(self, train_loader):
        self.vae_model.train()
        scaler = GradScaler()
        for _, (features, weights, _) in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            x = features.to(self.device)
            with torch.autocast(device_type='cuda', dtype=self.tensor_type):
                x_recontsructed, z_mean, z_sigma = self.vae_model.forward(x)

            weights = self._get_class_weights(weights)
            weights.to(self.device)

            # Check if if model output is valid (no NaN values)
            self._validate_model_output(x_recontsructed)
            loss = self.loss_function(model_output=x_recontsructed, target=x, z_mean=z_mean, z_sigma=z_sigma, weights=weights)

            loss_value = loss.item()
            if self.neptune_run:
                self.neptune_run["train/training_loss"].append(loss_value)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

    def _validate(self, validation_loader):
        self.vae_model.eval()
        epoch_val_loss = 0.0
        iterations = 0
        for _, (features, weights, _) in enumerate(validation_loader):
            x = features.to(self.device)

            with torch.autocast(device_type='cuda', dtype=self.tensor_type):
                x_recontsructed, z_mean, z_sigma = self.vae_model.forward(x)
            # Check if if model output is valid (no NaN values)
            self._validate_model_output(x_recontsructed)

            weights = self._get_class_weights(weights)
            weights.to(self.device)
            
            loss = self.loss_function(model_output=x_recontsructed, target=x, z_mean=z_mean, z_sigma=z_sigma, weights=weights)
            self.validation_logger.log_epoch_metrics(model_output=x_recontsructed, target=x, z_mean=z_mean, z_sigma=z_sigma, weights=weights)

            epoch_val_loss += loss.item()
            iterations += 1
        
        # Log the validation loss and validation metrics to neptune (if initialized)
        if self.neptune_run:
            self.neptune_run["train/validation_loss"].append(epoch_val_loss / iterations)

        # Log the validation metrics to neptune
        self.validation_logger.log_validation_metrics()

        return epoch_val_loss / iterations


    def inference(self, data_loader):
        model_inference_data = []
        cell_index_list = []

        vae_best_epoch_model = self._get_best_epoch_model()
        for _, (features, _, cell_indicies) in enumerate(data_loader):
            with torch.no_grad():
                x = features.to(self.device)
                with torch.autocast(device_type='cuda', dtype=self.tensor_type):
                    _, z_mean, z_sigma = vae_best_epoch_model.forward(x)
                    z_vector = vae_best_epoch_model.construct_z(z_mean, z_sigma)
            
            model_inference_data.append(z_vector.cpu().detach().numpy())
            cell_index_list.extend(cell_indicies)
        
        return model_inference_data, cell_index_list


    def plot_model_reconstruction(self, test_loader, post_process_function, dataset_name=None, cmap='gray', scaling_factor=255, post_process_function_model=None):
        test_data_iterator = iter(test_loader)
        test_batch = next(test_data_iterator)
        example_images = test_batch[0].to(self.device)

        vae_best_epoch_model = self._get_best_epoch_model()

        # Generate model output
        with torch.autocast(device_type='cuda', dtype=self.tensor_type):
            vae_model_output, _, _ = vae_best_epoch_model(example_images)
        
        reconstruction_fig = visualize_model_reconstruction(
            vae_model_output,
            example_images,
            post_process_function,
            cmap,
            scaling_factor,
            post_process_function_model
        )
    
        if self.neptune_run:
            self.neptune_run[f"{dataset_name}_reconstruction_example"].upload(reconstruction_fig)
