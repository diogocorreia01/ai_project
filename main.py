import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from ai_models.stable_diffusion import StableDiffusionAI
#from ai_models.stylegan import StyleGANAI
from ai_models.style_transfer import StyleTransferAI
import time
import platform
import torch
import threading

class AIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Generation and Style Transfer")
        self.root.geometry("800x550")

        # Label for the title
        self.label = tk.Label(self.root, text="Choose the type of AI you want to use:", font=("Arial", 21))
        self.label.pack(pady=20)

        # Button for Stable Diffusion
        self.button_sd = tk.Button(self.root, text="1 - Generate an image with Stable Diffusion", width=60, height=3, command=self.run_stable_diffusion)
        self.button_sd.pack(pady=20)

        # Button for StyleGAN
        self.button_sg = tk.Button(self.root, text="2 - Create an avatar with StyleGAN", width=60, height=3, command=self.run_stylegan)
        self.button_sg.pack(pady=20)

        # Button for Style Transfer
        self.button_st = tk.Button(self.root, text="3 - Apply Style Transfer", width=60, height=3, command=self.run_style_transfer)
        self.button_st.pack(pady=20)

        # Button to show system information
        self.button_sys_info = tk.Button(self.root, text="4 - Show System Information", width=60, height=3, command=self.show_system_info)
        self.button_sys_info.pack(pady=20)

    def run_stable_diffusion(self):
        # Get user input for the prompt
        prompt = self.ask_for_prompt()
        if prompt:
            # Create the loading pop-up with the spinning wheel
            self.show_loading_popup()

            # Create StableDiffusionAI instance and start a new thread for image generation
            sd = StableDiffusionAI()
            threading.Thread(target=self.generate_image_with_loading, args=(sd, prompt)).start()

    def run_stylegan(self):
        # Placeholder function for StyleGAN (not implemented yet)
        messagebox.showinfo("Coming Soon", "StyleGAN avatar generation is not implemented yet.")

    def run_style_transfer(self):
        # Apply style transfer
        #content_path = "assets/img_teste.jpg"
        #style_path = "assets/estilo_pintura.jpg"

        # Inform user that processing has started
        #self.show_processing_message()

        # Simulate the processing time (this will be replaced with actual style transfer logic)
        #for i in range(101):
            #time.sleep(0.05)  # Simulating processing delay
            #self.update_progress(i)  # Update the progress bar

        #st = StyleTransferAI()
        #st.apply_style(content_path, style_path)
        #messagebox.showinfo("Success", "Style transfer applied successfully!")
        # Placeholder function for style_transfer (not implemented yet)
        messagebox.showinfo("Coming Soon", "Style Transfer generation is not implemented yet.")

    def ask_for_prompt(self):
        # Create a simple input dialog to get the prompt
        prompt = simpledialog.askstring("Input", "Enter the prompt to generate the image:")
        return prompt

    def show_loading_popup(self):
        """ Create and show a pop-up window with a loading spinner. """
        self.loading_popup = tk.Toplevel(self.root)
        self.loading_popup.title("Processing...")
        self.loading_popup.geometry("300x150")  # Size of the pop-up

        # Create a label with spinning wheel
        self.loading_label = tk.Label(self.loading_popup, text="Image is being processed...\nPlease wait.",
                                      font=("Arial", 12))
        self.loading_label.pack(pady=20)

        # Add a spinning wheel (animated gif or placeholder)
        self.spinner = ttk.Progressbar(self.loading_popup, orient="horizontal", length=200, mode="indeterminate")
        self.spinner.pack(pady=20)
        self.spinner.start()  # Start spinning animation

    def hide_loading_popup(self):
        """ Close the loading pop-up window. """
        self.spinner.stop()  # Stop spinning animation
        self.loading_popup.destroy()

    def generate_image_with_loading(self, sd, prompt):
        """ Generate the image using StableDiffusionAI while showing the loading popup. """
        output_path = sd.generate_image(prompt)

        # Close the loading pop-up window after image generation
        self.hide_loading_popup()

        # Inform the user that the image generation is done
        messagebox.showinfo("Success", f"Image generated successfully!\nSaved at: {output_path}")

    def show_system_info(self):
        """ Display system information including CUDA support. """
        system_info = f"System: {platform.system()}\n"
        system_info += f"Machine: {platform.machine()}\n"
        system_info += f"Processor: {platform.processor()}\n"
        system_info += f"Python Version: {platform.python_version()}\n"
        system_info += f"CUDA Available: {'Yes' if torch.cuda.is_available() else 'No'}\n"

        # Show system info in a pop-up window
        messagebox.showinfo("System Information", system_info)

def main():
    root = tk.Tk()
    app = AIApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
