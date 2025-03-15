from ai_models.stable_diffusion import StableDiffusionAI
#from ai_models.stylegan import StyleGANAI
from ai_models.style_transfer import StyleTransferAI

def main():
    print("Choose the type of AI you want to use:")
    print("1 - Generate an image with Stable Diffusion")
    print("2 - Create an avatar with StyleGAN")
    print("3 - Apply Style Transfer")

    choice = input("Enter the number of your option: ")

    if choice == "1":
        sd = StableDiffusionAI()
        prompt = input("Enter the prompt to generate the image: ")
        sd.generate_image(prompt)

    elif choice == "2":
        #sg = StyleGANAI()
        #sg.generate_avatar()
        pass

    elif choice == "3":
        st = StyleTransferAI()
        content_path = "assets/img_teste.jpg"
        style_path = "assets/estilo_pintura.jpg"
        st.apply_style(content_path, style_path)

    else:
        print("Invalid option!")

if __name__ == "__main__":
    main()
