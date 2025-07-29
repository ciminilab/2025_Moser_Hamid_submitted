import numpy as np
import matplotlib.pyplot as plt 

def visualize_model_reconstruction(model_output, example_images, post_process_function, cmap='gray', scaling_factor=255, post_process_function_model=None):
    '''
    Visualize the model reconstruction with four examples.
            Parameters:
                    x (tensor): Input tensor x
                    x_recontsructed (tensor): Reconstructed input x
    '''
    fig, axes = plt.subplots(2, 4, figsize=(10, 4))

    x = post_process_function(example_images)
    if post_process_function_model:
        x_reconstructed = post_process_function_model(model_output)
    else:
        x_reconstructed = post_process_function(model_output)

    for i in range(4):
        img = x[i].cpu().detach().numpy()
        img_recon = x_reconstructed[i].cpu().detach().numpy()

        uint8_image = (img * scaling_factor).round().astype(np.uint8)
        uint8_image_recon = (np.squeeze(img_recon) * scaling_factor).round().astype(np.uint8)

        axes[0, i].imshow(uint8_image, cmap=cmap)
        axes[0, i].axis('off')

        axes[1, i].imshow(uint8_image_recon, cmap=cmap)
        axes[1, i].axis('off')

    fig.suptitle('Reconstruction examples', fontsize=16)
    return fig
