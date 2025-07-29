import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from monai.losses import DiceFocalLoss, DiceLoss, FocalLoss
from .helper_functions import kld_loss_function, dice_score_function, bce_loss_function

class LossFunction(_Loss):
    def __init__(self, beta=1) -> None:
        super().__init__()
        self.beta = beta

class MSELossFunction(LossFunction):
    def __init__(self, reduction='mean', beta=1, logits=False) -> None:
        super().__init__(beta)

    def forward(self, model_output, target, z_sigma, z_mean, weights):
        return nn.MSELoss()(model_output, target)

class HuberLossFunction(LossFunction):
    def __init__(self, reduction='mean', beta=1, logits=False) -> None:
        super().__init__(beta)

    def forward(self, model_output, target, z_sigma, z_mean, weights):
        return nn.HuberLoss()(model_output, target)

class VAELossFunction(LossFunction):
    def __init__(self, reduction='mean', beta=1, logits=False) -> None:
        super().__init__(beta)
        self.reduction = reduction
        self.logits = logits

    def forward(self, model_output, target, z_sigma, z_mean, weights) -> None:
        reconstruction_loss = bce_loss_function(model_output, target, self.logits, self.reduction)

        kld_loss = kld_loss_function(z_mean, z_sigma)

        return reconstruction_loss + self.beta * kld_loss

class VAEDiceLossFunction(LossFunction):
    def __init__(self, reduction='mean', beta=1, logits=False) -> None:
        super().__init__(beta)

    def forward(self, model_output, target, z_sigma, z_mean, weights) -> None:
        dice_loss = 1 - dice_score_function(model_output, target).mean()

        kld_loss = kld_loss_function(z_mean, z_sigma)

        return dice_loss + self.beta * kld_loss

class BCEDiceLossFunction(LossFunction):
    def __init__(self, reduction='mean', beta=1, logits=False) -> None:
        """
        Loss function class which caluclates the sum of BCE loss and the Dice loss.
        The loss terms are not weighted. NOTE: This loss function expectes logits.
        """
        super().__init__(beta)
        self.logits = logits

    def forward(self,  model_output, target, z_sigma, z_mean, weights):
        kld_loss = kld_loss_function(z_mean, z_sigma)

        reconstruction_loss = bce_loss_function(model_output, target, self.logits, self.reduction)

        model_output = torch.sigmoid(model_output)
        dice_loss = 1 - dice_score_function(model_output, target).mean()

        return reconstruction_loss + dice_loss + self.beta * kld_loss

class MSEDiceLossFunction(LossFunction):
    def __init__(self, reduction='mean', beta=1, logits=False) -> None:
        """
        Loss function class which caluclates the sum of MSE loss and the Dice loss.
        The loss terms are not weighted. NOTE: This loss function expectes boths output and target to be in [0, 1].
        """
        super().__init__(beta)
        self.reduction = reduction

    def forward(self,  model_output, target, z_sigma, z_mean, weights):
        kld_loss = kld_loss_function(z_mean, z_sigma)

        mse_loss = nn.MSELoss(reduction=self.reduction)(model_output, target)

        dice_loss = 1 - dice_score_function(model_output, target).mean()

        return mse_loss + dice_loss + self.beta * kld_loss

class DiceFocalLossFunction(LossFunction):
    def __init__(self, reduction='mean', beta=1, logits=False) -> None:
        super().__init__(beta)

    def forward(self,  model_output, target, z_sigma, z_mean, weights):        
        focal_dice_loss_fun = DiceFocalLoss(squared_pred=False, smooth_dr=0.01, weight=weights, lambda_focal=0.2)
        focal_dice_loss = focal_dice_loss_fun(model_output, target)

        return focal_dice_loss

class DiceFocalGenecountLossFunction(LossFunction):
    def __init__(self, reduction='mean', beta=1, logits=False) -> None:
        super().__init__(beta)

    def forward(self,  model_output, target, z_sigma, z_mean, weights):        
        kld_loss = kld_loss_function(z_mean, z_sigma)

        focal_dice_loss_fun = DiceFocalLoss(squared_pred=False, smooth_dr=0.01, weight=weights, lambda_focal=0.2)
        focal_dice_loss = focal_dice_loss_fun(model_output, target)

        image_width = model_output.shape[-1]

        gene_count_estimates = torch.sum(model_output, dim=(2, 3)) / (image_width * image_width)
        gene_count_target = torch.sum(target, dim=(2, 3)) / (image_width * image_width)
        gene_count_loss = nn.MSELoss(reduction=self.reduction)(gene_count_estimates, gene_count_target)

        return focal_dice_loss + gene_count_loss + self.beta * kld_loss

class L2ReconstructionLossFunction(LossFunction):
    def __init__(self, reduction='mean', beta=1, logits=False) -> None:
        """
        Calculates the L2 loss between model reconstruction and target.
        """
        super().__init__(beta)
        self.logits = logits

    def forward(self,  model_output, target, z_sigma, z_mean, weights):
        kld_loss = kld_loss_function(z_mean, z_sigma)

        reconstruction_loss = nn.MSELoss(reduction=self.reduction)(model_output, target)

        return reconstruction_loss + self.beta * kld_loss

class DoubleEncLossFunction(LossFunction):
    def __init__(self, reduction='mean', beta=1, logits=False) -> None:
        super().__init__(beta)
        self.reduction = reduction

    def forward(self,  model_output, target, z_sigma, z_mean, weights):
        kld_loss = kld_loss_function(z_mean, z_sigma)

        reconstruction_loss_one = nn.MSELoss(reduction=self.reduction)(model_output[0], target[:, :115])
        reconstruction_loss_two = nn.MSELoss(reduction=self.reduction)(model_output[1], target[:, 115:])

        return reconstruction_loss_one + reconstruction_loss_two + self.beta * kld_loss

class CombinedSingleCropTranscriptsLossFunction(LossFunction):
    def __init__(self, reduction='mean', beta=1, logits=False) -> None:
        super().__init__(beta)
        self.reduction = reduction

    def forward(self,  model_output, target, z_sigma, z_mean, weights):
        kld_loss = kld_loss_function(z_mean, z_sigma)

        # First channel contains the staining crop
        crop_reconstruction_loss = nn.MSELoss(reduction=self.reduction)(model_output[:, 0], target[:, 0])
        weights = torch.cat((torch.tensor([0]), torch.ones_like(weights)), dim=0)

        # The remaining channels are the transcripts
        focal_dice_loss_fun = DiceFocalLoss(squared_pred=False, smooth_dr=0.01, weight=weights, lambda_focal=0.4, lambda_dice=0.8)
        focal_dice_loss = focal_dice_loss_fun(model_output, target)

        return focal_dice_loss + self.beta * kld_loss + crop_reconstruction_loss

class DoubleDecoderLossFunction(LossFunction):
    def __init__(self, reduction='mean', beta=1, logits=False) -> None:
        super().__init__(beta)
        self.reduction = reduction

    def forward(self,  model_output, target, z_sigma, z_mean, weights):
        kld_loss = kld_loss_function(z_mean, z_sigma)

        crop_reconstruction_loss = nn.MSELoss(reduction=self.reduction)(model_output[1], target[:, :2])

        # The remaining channels are the transcripts
        focal_dice_loss_fun = DiceFocalLoss(squared_pred=False, smooth_dr=0.01, lambda_focal=0.4, lambda_dice=0.8)
        focal_dice_loss = focal_dice_loss_fun(model_output[0], target[:, 2:])

        return focal_dice_loss + self.beta * kld_loss + crop_reconstruction_loss
