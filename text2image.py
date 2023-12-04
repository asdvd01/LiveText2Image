from diffusers import AutoPipelineForText2Image
import torch
import gradio as gr

def text2image(prompt):
    """
    This function takes a text prompt as input and returns an image.

    Parameters:
    prompt (str): The text prompt to generate the image from.

    Returns:
    image: The generated image.
    """

    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("mps")     #Tested on Intel Macbook Pro 16" with 16GB RAM. 
    #Run this file on Intel Macbook using   `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python text2image.py` 
    #pipe.to("cuda")   #Use this if you have a GPU
    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    return image

interface = gr.Interface(fn=text2image, inputs="text", outputs="image") #TODO: gr.Interface(fn=text2image, inputs="text", outputs="image", live=True)
interface.launch()


