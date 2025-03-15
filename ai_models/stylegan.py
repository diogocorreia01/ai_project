import torch
from stylegan2_pytorch import Generator # Error
from PIL import Image
import numpy as np
import uuid
import os

class StyleGANAI:
    def __init__(self, model_path="path_to_trained_model.pth"):
        # Load the pre-trained model
        self.generator = Generator.load(model_path)

        # Create output folder if it doesn't exist
        os.makedirs("results", exist_ok=True)

    def generate_avatar(self):
        # Generate a random latent vector
        latent_vector = torch.randn((1, 512))

        # Generate the image using the generator
        with torch.no_grad():
            generated_image = self.generator(latent_vector)

        # Convert the tensor to a NumPy array, then to a PIL image
        generated_image = generated_image.squeeze().cpu().numpy().transpose(1, 2, 0)
        generated_image = np.clip(generated_image, 0, 1)  # Ensure valid pixel range [0, 1]
        pil_image = Image.fromarray((generated_image * 255).astype(np.uint8))

        # Generate a random ID for the image filename
        image_id = str(uuid.uuid4())[:8]  # Use the first 8 characters of UUID
        output_path = f"results/avatar_generated_{image_id}.png"

        # Save and display the image
        pil_image.save(output_path)
        pil_image.show()

        print(f"Avatar image saved at: {output_path}")
        return output_path
