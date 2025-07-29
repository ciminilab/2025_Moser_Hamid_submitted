import torch
import torch.nn as nn
from monai.losses import DiceLoss
from .helper_functions import dice_score_function, kld_loss_function

class BaseValidationLogger():
    """
    Base class validation logger witout any effects.
    """
    def __init__(self) -> None:
        pass

    def log_epoch_metrics(self, model_output, target, z_mean, z_sigma, weights):
        pass

    def log_validation_metrics(self):
        pass

class SegmentationValidationLogger():
    def __init__(self, neptune_run, logit_output=False) -> None:
        self.epoch_val_loss, self.epoch_dice_loss, self.epoch_bce_loss, self.epoch_kdl_loss = 0.0, 0.0, 0.0, 0.0
        self.iterations = 0
        self.neptune_run = neptune_run
        self.logit_output = logit_output
        
    def log_epoch_metrics(self, model_output, target, z_mean, z_sigma, weights):
        kld_loss = kld_loss_function(z_mean, z_sigma)

        if self.logit_output:
            model_output = torch.sigmoid(model_output)
            bce_loss = nn.BCEWithLogitsLoss(reduction='mean')(model_output, target)
        else:
            bce_loss = nn.BCELoss(reduction='mean')(model_output, target)

        binary_weights = torch.min(weights, torch.ones(weights.shape))
        binary_model_output = torch.round(model_output)
        dice_loss_function = DiceLoss(squared_pred=False, weight=binary_weights)
        dice_loss = dice_loss_function(binary_model_output, target)

        self.epoch_dice_loss += dice_loss.item()
        self.epoch_bce_loss += bce_loss.item()
        self.epoch_kdl_loss += kld_loss.item()
        self.iterations += 1

    def log_validation_metrics(self):
        # Log the validation loss and validation metrics to neptune (if initialized)
        if self.neptune_run:
            self.neptune_run["train/validation_dice_loss"].append(self.epoch_dice_loss / self.iterations)
            self.neptune_run["train/validation_bce_metric"].append(self.epoch_bce_loss / self.iterations)
            self.neptune_run["train/validation_kdl_metric"].append(self.epoch_kdl_loss / self.iterations)

class CombinedSingleCropTranscriptsValidationLogger():
    def __init__(self, neptune_run, logit_output=False) -> None:
        self.epoch_val_loss, self.epoch_dice_loss, self.crop_reconstruction_loss, self.epoch_kdl_loss, self.epoch_bce_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        self.iterations = 0
        self.neptune_run = neptune_run
        self.logit_output = logit_output
        
    def log_epoch_metrics(self, model_output, target, z_mean, z_sigma, weights):
        kld_loss = kld_loss_function(z_mean, z_sigma)

        if self.logit_output:
            model_output = torch.sigmoid(model_output)
            bce_loss = nn.BCEWithLogitsLoss(reduction='mean')(model_output[:, 1:], target[:, 1:])
        else:
            bce_loss = nn.BCELoss(reduction='mean')(model_output[:, 1:], target[:, 1:])

        self.crop_reconstruction_loss += nn.MSELoss(reduction='mean')(model_output[:, 0], target[:, 0]).item()

        binary_model_output = torch.round(model_output)
        dice_loss_function = DiceLoss(squared_pred=False)
        dice_loss = dice_loss_function(binary_model_output[:, 1:], target[:, 1:])

        self.epoch_dice_loss += dice_loss.item()
        self.epoch_bce_loss += bce_loss.item()
        self.epoch_kdl_loss += kld_loss.item()
        self.iterations += 1

    def log_validation_metrics(self):
        # Log the validation loss and validation metrics to neptune (if initialized)
        if self.neptune_run:
            self.neptune_run["train/validation_dice_loss"].append(self.epoch_dice_loss / self.iterations)
            self.neptune_run["train/validation_crop_reconstruction_loss"].append(self.crop_reconstruction_loss / self.iterations)
            self.neptune_run["train/validation_kdl_metric"].append(self.epoch_kdl_loss / self.iterations)
            self.neptune_run["train/validation_bce_metric"].append(self.epoch_bce_loss / self.iterations)

class DoubleDecoderCombinedSingleCropTranscriptsValidationLogger():
    def __init__(self, neptune_run, logit_output=False) -> None:
        self.epoch_val_loss, self.epoch_dice_loss, self.crop_reconstruction_loss, self.epoch_kdl_loss, self.epoch_bce_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        self.iterations = 0
        self.neptune_run = neptune_run
        self.logit_output = logit_output
        
    def log_epoch_metrics(self, model_output, target, z_mean, z_sigma, weights):
        kld_loss = kld_loss_function(z_mean, z_sigma)

        bce_loss = nn.BCELoss(reduction='mean')(model_output[0], target[:, 2:])

        binary_model_output = torch.round(model_output[0])
        dice_loss_function = DiceLoss(squared_pred=False)
        dice_loss = dice_loss_function(binary_model_output, target[:, 2:])

        self.crop_reconstruction_loss += nn.MSELoss(reduction='mean')(model_output[1], target[:, :2]).item()

        self.epoch_dice_loss += dice_loss.item()
        self.epoch_bce_loss += bce_loss.item()
        self.epoch_kdl_loss += kld_loss.item()
        self.iterations += 1

    def log_validation_metrics(self):
        # Log the validation loss and validation metrics to neptune (if initialized)
        if self.neptune_run:
            self.neptune_run["train/validation_dice_loss"].append(self.epoch_dice_loss / self.iterations)
            self.neptune_run["train/validation_crop_reconstruction_loss"].append(self.crop_reconstruction_loss / self.iterations)
            self.neptune_run["train/validation_kdl_metric"].append(self.epoch_kdl_loss / self.iterations)
            self.neptune_run["train/validation_bce_metric"].append(self.epoch_bce_loss / self.iterations)

