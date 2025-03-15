from diffusers import StableDiffusionXLPipeline
import torch
import uuid
import os
import time

class StableDiffusionAI:
    def __init__(self):
        print("Loading Stable Diffusion XL model...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        self.pipe.to("cuda")  # Move model to GPU

        # Create output folder if it does not exist
        os.makedirs("results", exist_ok=True)

    def generate_image(self, prompt):
        # Generate a random ID for the image filename
        image_id = str(uuid.uuid4())[:8]  # Use first 8 characters
        output_path = f"results/image_{image_id}.png"

        print(f"Generating image for prompt: {prompt}")

        # Simulate the image generation with progress (for illustration)
        steps = 50  # Adjust to match the number of processing steps
        for i in range(steps):
            time.sleep(0.1)  # Simulate the processing time per step

        # Generate image (simulate the actual processing)
        image = self.pipe(prompt).images[0]

        # Save and display the image
        image.save(output_path)
        image.show()

        print(f"Image saved at: {output_path}")
        return output_path
