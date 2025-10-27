import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fire import Fire

from s3od import BackgroundRemoval
from s3od.visualizer import visualize_removal

# Model variants mapping
MODEL_VARIANTS = {
    'General (Synth + Real)': 'okupyn/s3od',
    'Synthetic Only': 'okupyn/s3od-synth',
    'DIS-tuned': 'okupyn/s3od-dis',
    'SOD-tuned': 'okupyn/s3od-sod',
}

# Cache loaded models to avoid reloading
_model_cache = {}

def get_detector(model_name):
    """Get or load detector for the specified model."""
    if model_name not in _model_cache:
        print(f"Loading model: {model_name}")
        _model_cache[model_name] = BackgroundRemoval(model_id=model_name)
    return _model_cache[model_name]

# Load default model
detector = get_detector('okupyn/s3od')

VISUALIZATION_METHODS = {
    'Transparent Background': 'transparent',
    'White Background': 'white',
    'Green Background': 'green',
    'Mask Only': 'mask'
}


def compute_mask_iou(mask1, mask2):
    """Compute IoU between two masks."""
    intersection = np.logical_and(mask1 > 0.5, mask2 > 0.5).sum()
    union = np.logical_or(mask1 > 0.5, mask2 > 0.5).sum()
    return intersection / (union + 1e-6)


def is_ambiguous(all_masks, threshold=0.8):
    """Check if prediction is ambiguous based on mask IoU."""
    if len(all_masks) < 2:
        return False
    
    # Compute IoU between all pairs
    for i in range(len(all_masks)):
        for j in range(i + 1, len(all_masks)):
            iou = compute_mask_iou(all_masks[i], all_masks[j])
            if iou < threshold:
                return True
    return False


def create_masks_grid(all_masks, all_ious, image_shape):
    """Create a grid showing all 3 masks side by side."""
    h, w = image_shape[:2]
    num_masks = len(all_masks)
    
    # Create grid image
    grid_w = w * num_masks
    grid_h = h
    grid = Image.new('L', (grid_w, grid_h), color=0)
    
    for idx, (mask, iou) in enumerate(zip(all_masks, all_ious)):
        # Convert mask to image
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        
        # Paste into grid
        grid.paste(mask_img, (idx * w, 0))
    
    return grid


def process_image(image, model_key, method_key, threshold):
    if image is None:
        return None, None, None
    
    # Get the appropriate model
    model_id = MODEL_VARIANTS.get(model_key, 'okupyn/s3od')
    detector = get_detector(model_id)
    
    result = detector.remove_background(image, threshold=threshold)
    method = VISUALIZATION_METHODS.get(method_key, 'transparent')
    
    # Generate main output
    if method == 'transparent':
        main_output = result.rgba_image
    elif method == 'white':
        main_output = visualize_removal(image, result, background_color=(255, 255, 255))
    elif method == 'green':
        main_output = visualize_removal(image, result, background_color=(0, 255, 0))
    elif method == 'mask':
        mask_vis = (result.predicted_mask * 255).astype(np.uint8)
        main_output = Image.fromarray(mask_vis, mode='L')
    else:
        main_output = result.rgba_image
    
    # Create masks grid
    masks_grid = create_masks_grid(result.all_masks, result.all_ious, image.shape)
    
    # Check if ambiguous
    ambiguous = is_ambiguous(result.all_masks)
    ambiguity_label = "⚠️ Ambiguous prediction (IoU < 0.8 between masks)" if ambiguous else "✓ Clear prediction"
    
    return main_output, masks_grid, ambiguity_label


with gr.Blocks(title="S3OD - Synthetic Salient Object Detection") as demo:
    gr.Markdown("""
    # S3OD: Synthetic Salient Object Detection
    
    Upload an image to remove its background using **S3OD**! 
    
    S3OD is trained on a large-scale fully synthetic dataset (140K+ images) generated with diffusion models. 
    The model uses a DPT-based architecture with DINOv3 vision transformer backbone for robust salient object detection.
    
    **Model Variants:**
    - **General (Synth + Real)**: Default model trained on synthetic data and fine-tuned on all real datasets (DUTS, DIS, HR-SOD)
    - **Synthetic Only**: Trained exclusively on S3OD synthetic dataset
    - **DIS-tuned**: Fine-tuned specifically for highly-accurate dichotomous segmentation
    - **SOD-tuned**: Optimized for general salient object detection tasks
    
    **Key Features:**
    - Single-step background removal with soft masks (smooth edges)
    - Multi-mask prediction with IoU scoring
    - Ambiguity detection for uncertain predictions
    - Works on any image resolution
    
    📄 [Paper](https://arxiv.org/abs/2510.21605) | 💻 [GitHub](https://github.com/KupynOrest/s3od) | 🤗 [Model](https://huggingface.co/okupyn/s3od) | 🗂️ [Dataset](https://huggingface.co/datasets/okupyn/s3od_dataset)
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Upload an Image")
            model_dropdown = gr.Dropdown(
                choices=list(MODEL_VARIANTS.keys()),
                label="Model Variant",
                value='General (Synth + Real)',
                info="Choose the model variant trained on different datasets"
            )
            method_radio = gr.Radio(
                list(VISUALIZATION_METHODS.keys()),
                label="Output Format",
                value='Transparent Background'
            )
            threshold_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.05,
                label="Mask Threshold"
            )
            submit_btn = gr.Button("Remove Background", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(type="pil", label="Result")
            ambiguity_label = gr.Textbox(label="Prediction Quality", interactive=False)
    
    with gr.Row():
        masks_grid = gr.Image(type="pil", label="All 3 Predicted Masks (with IoU scores)")
    
    submit_btn.click(
        fn=process_image,
        inputs=[input_image, model_dropdown, method_radio, threshold_slider],
        outputs=[output_image, masks_grid, ambiguity_label]
    )
    
    # Also trigger on image upload
    input_image.change(
        fn=process_image,
        inputs=[input_image, model_dropdown, method_radio, threshold_slider],
        outputs=[output_image, masks_grid, ambiguity_label]
    )


def main(server_name="0.0.0.0", server_port=7860, share=False):
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share
    )


if __name__ == '__main__':
    Fire(main)

