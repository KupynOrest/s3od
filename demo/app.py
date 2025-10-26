import gradio as gr
import numpy as np
from PIL import Image
from fire import Fire

from s3od import BackgroundRemoval
from s3od.visualizer import visualize_removal

detector = BackgroundRemoval()

VISUALIZATION_METHODS = {
    'Transparent Background': 'transparent',
    'White Background': 'white',
    'Green Background': 'green',
    'Mask Only': 'mask'
}


def process_image(image, method, threshold):
    if image is None:
        return None
    
    result = detector.remove_background(image, threshold=threshold)
    
    if method == 'transparent':
        return result.rgba_image
    elif method == 'white':
        return visualize_removal(image, result, background_color=(255, 255, 255))
    elif method == 'green':
        return visualize_removal(image, result, background_color=(0, 255, 0))
    elif method == 'mask':
        mask_vis = (result.predicted_mask * 255).astype(np.uint8)
        return Image.fromarray(mask_vis, mode='L')
    
    return result.rgba_image


iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", label="Upload an Image"),
        gr.Radio(
            list(VISUALIZATION_METHODS.keys()),
            label="Output Format",
            value='Transparent Background'
        ),
        gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.5,
            step=0.05,
            label="Mask Threshold"
        )
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="Demo: S3OD - Synthetic Salient Object Detection",
    description="""
    Upload an image to remove its background using **S3OD**! 
    
    S3OD is trained on a large-scale fully synthetic dataset (140K+ images) generated with diffusion models. 
    Despite being trained only on synthetic data, it achieves state-of-the-art performance on real-world images.
    
    The model uses a DPT-based architecture with DINOv3 vision transformer backbone for robust salient object detection 
    and can process images of any size. Choose from four visualization methods: transparent background (RGBA), 
    white background, green background (chroma key), or mask only.
    
    **Key Features:**
    - Single-step background removal
    - Multi-mask prediction with IoU scoring
    - Adjustable threshold for fine-tuning
    - Works on any image resolution
    
    Ideal for applications in e-commerce, content creation, photo editing, and computer vision research.
    
    📄 [Paper](https://arxiv.org/abs/XXXX.XXXXX) | 💻 [GitHub](https://github.com/KupynOrest/s3od) | 🤗 [Model](https://huggingface.co/okupyn/s3od) | 🗂️ [Dataset](https://huggingface.co/datasets/okupyn/s3od_dataset)
    """,
    allow_flagging='never',
    examples=[
        # Add example images here when available
    ]
)


def main(server_name="0.0.0.0", server_port=7860, share=False):
    iface.launch(
        server_name=server_name,
        server_port=server_port,
        share=share
    )


if __name__ == '__main__':
    Fire(main)

