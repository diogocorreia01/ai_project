import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import uuid
import os
from PIL import Image

class StyleTransferAI:
    def __init__(self):
        print("Loading Style Transfer model...")
        self.model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

        # Create output directory if it does not exist
        os.makedirs("results", exist_ok=True)

    def load_image(self, img_path, target_size=(256, 256)):
        """Loads and preprocesses an image for style transfer."""
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def apply_style(self, content_path, style_path):
        """Applies style transfer and saves the image with a random filename."""
        print(f"Applying style transfer to: {content_path} with style: {style_path}")

        content_image = self.load_image(content_path)
        style_image = self.load_image(style_path)
        stylized_image = self.model(tf.constant(content_image), tf.constant(style_image))[0]

        # Generate a random filename
        image_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
        output_path = f"results/stylized_image_{image_id}.png"

        # Convert to uint8 and save the image
        stylized_image = np.array(stylized_image[0] * 255, dtype=np.uint8)
        Image.fromarray(stylized_image).save(output_path)
        Image.fromarray(stylized_image).show()

        print(f"Stylized image saved at: {output_path}")
        return output_path
