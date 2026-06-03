import numpy as np
import cv2
from PIL import Image
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import json
import logging
from typing import Optional, Dict, Any

from ..filter_dataset import BaseFilter, FilterResult, Sample


class GemmaSemanticFilter(BaseFilter):
    """Filter using Gemma-3 VLM to evaluate semantic quality (salient object presence and coverage)."""
    
    def __init__(self, 
                 model_id: str = "google/gemma-3-4b-it",
                 name: str = "semantic_quality",
                 confidence_threshold: float = 0.7,
                 target_size: int = 672,
                 device_map: str = "auto"):
        super().__init__(name)
        
        self.confidence_threshold = confidence_threshold
        self.model_id = model_id
        self.target_size = target_size  # Optimal size for Gemma-3 VLM
        
        self.model = None
        self.processor = None
        self._model_loaded = False
        
        logging.info(f"GemmaSemanticFilter initialized with model: {model_id}, target_size: {target_size}")
    
    def _load_model(self):
        """Lazy load the Gemma model to save memory."""
        if self._model_loaded:
            return
            
        try:
            logging.info(f"Loading Gemma model: {self.model_id}")
            
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            ).eval()
            
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            self._model_loaded = True
            logging.info("Gemma model loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to load Gemma model: {e}")
            raise
    
    def _resize_for_vlm(self, image: Image.Image) -> Image.Image:
        """Resize image to optimal resolution for VLM inference."""
        # Get current size
        width, height = image.size
        
        # Calculate aspect ratio preserving resize
        aspect_ratio = width / height
        
        if aspect_ratio > 1:
            # Landscape: limit width
            new_width = self.target_size
            new_height = int(self.target_size / aspect_ratio)
        else:
            # Portrait: limit height  
            new_height = self.target_size
            new_width = int(self.target_size * aspect_ratio)
        
        # Resize using high-quality resampling
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        logging.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        return resized_image
    
    def _create_semantic_visualization(self, image: np.ndarray, mask: np.ndarray) -> Image.Image:
        """Create visualization focusing on semantic content: image | overlay."""
        
        # Ensure mask is single channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Create overlay (image with red mask overlay)
        overlay = image.copy()
        mask_norm = mask.astype(np.float32) / 255.0
        red_overlay = np.zeros_like(image)
        red_overlay[:, :, 0] = mask_norm * 255  # Red channel
        overlay = (overlay * 0.7 + red_overlay * 0.3).astype(np.uint8)
        
        # Stack horizontally: [image | overlay]
        stacked = np.hstack([image, overlay])
        
        # Convert to PIL and resize for optimal VLM processing
        pil_image = Image.fromarray(stacked)
        return self._resize_for_vlm(pil_image)
    
    def _get_semantic_prompt(self) -> str:
        """Get evaluation prompt focused on semantic quality."""
        return """You are an expert computer vision specialist evaluating image segmentation for semantic correctness.

TASK: Analyze the 2-panel image:
- LEFT: Original image  
- RIGHT: Overlay showing segmentation mask in red

Respond with ONLY this JSON format:
{
  "has_salient_object": true/false,
  "covers_object": true/false,
  "confidence": 0.0-1.0
}

EVALUATION CRITERIA:

1. has_salient_object: Is there a clear, distinct main object that should be segmented?
   - TRUE: Clear foreground objects (people, animals, vehicles, tools, furniture, distinct items)
   - FALSE: Pure landscapes, textures, abstract patterns, empty backgrounds, unclear scenes

2. covers_object: Does the red overlay capture the main object adequately?
   - TRUE: Red area covers majority of main object (>70%), follows object boundaries reasonably
   - FALSE: Missing major object parts (>30%), captures mostly background, severely misaligned

FOCUS ON:
- Is there an obvious main subject in the scene?
- Does the red highlight represent that main subject reasonably well?
- Ignore fine details - focus on overall semantic correctness

PASS: Clear objects with reasonable coverage
FAIL: No main object OR poor object coverage"""

    def _evaluate_semantic(self, visualization: Image.Image) -> Dict[str, Any]:
        """Evaluate semantic quality using Gemma VLM."""
        
        if not self._model_loaded:
            self._load_model()
        
        system_prompt = self._get_semantic_prompt()
        
        messages = [
            {
                "role": "system", 
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": visualization},
                    {"type": "text", "text": "Evaluate semantic quality:"}
                ]
            }
        ]

        model_inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            model_inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in model_inputs.items()}

        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
            generation = generation[0][input_len:]

        response = self.processor.decode(generation, skip_special_tokens=True).strip()
        
        try:
            response = response.strip()
            if response.startswith('```json'):
                response = response.replace('```json', '').replace('```', '').strip()
            elif response.startswith('```'):
                response = response.replace('```', '').strip()
            
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                evaluation = json.loads(json_str)
            else:
                raise ValueError("No JSON object found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"Failed to parse semantic VLM response: {e}")
            response_lower = response.lower()
            has_salient = any(word in response_lower for word in ["true", "yes", "clear", "object"])
            covers_well = any(word in response_lower for word in ["good", "covers", "accurate"])
            
            evaluation = {
                "has_salient_object": has_salient,
                "covers_object": covers_well,
                "confidence": 0.4
            }
        
        return evaluation
    
    def filter(self, sample: Sample) -> FilterResult:
        """Evaluate semantic quality."""
        image = sample.load_image()
        mask = sample.load_mask(binary=False)
        
        visualization = self._create_semantic_visualization(image, mask)
        evaluation = self._evaluate_semantic(visualization)
        
        has_salient = evaluation.get("has_salient_object", False)
        covers_object = evaluation.get("covers_object", False)
        confidence = evaluation.get("confidence", 0.0)
        
        criteria_passed = has_salient and covers_object
        confidence_passed = confidence >= self.confidence_threshold
        final_passed = criteria_passed and confidence_passed
        
        if not final_passed:
            reasons = []
            if not has_salient:
                reasons.append("no_clear_salient_object")
            if not covers_object:
                reasons.append("poor_object_coverage")
            if not confidence_passed:
                reasons.append(f"low_confidence={confidence:.3f}")
            reason = ", ".join(reasons)
        else:
            reason = None
        
        problem_desc = ""
        if not has_salient:
            problem_desc += "No Clear Main Object"
        if not covers_object:
            problem_desc += " | Poor Coverage" if problem_desc else "Poor Coverage"
        if not confidence_passed:
            problem_desc += f" | Low Conf ({confidence:.2f})" if problem_desc else f"Low Conf ({confidence:.2f})"
        
        return FilterResult(
            passed=final_passed,
            reason=reason,
            score=confidence,
            metadata={
                'semantic_evaluation': evaluation,
                'problem_description': problem_desc or f"Semantic OK [conf: {confidence:.2f}]"
            }
        )


class GemmaMaskArtifactFilter(BaseFilter):
    """Filter using Gemma-3 VLM to detect mask artifacts and fragmentation."""
    
    def __init__(self, 
                 model_id: str = "google/gemma-3-4b-it",
                 name: str = "mask_artifacts",
                 confidence_threshold: float = 0.7,
                 target_size: int = 672):
        super().__init__(name)
        
        self.confidence_threshold = confidence_threshold
        self.model_id = model_id
        self.target_size = target_size  # Optimal size for Gemma-3 VLM
        
        self.model = None
        self.processor = None
        self._model_loaded = False
        
        logging.info(f"GemmaMaskArtifactFilter initialized with target_size: {target_size}")
    
    def _load_model(self):
        """Lazy load the Gemma model to save memory."""
        if self._model_loaded:
            return
            
        try:
            logging.info(f"Loading Gemma model: {self.model_id}")
            
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            ).eval()
            
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            self._model_loaded = True
            logging.info("Gemma model loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to load Gemma model: {e}")
            raise
    
    def _resize_for_vlm(self, image: Image.Image) -> Image.Image:
        """Resize image to optimal resolution for VLM inference."""
        width, height = image.size
        aspect_ratio = width / height
        
        if aspect_ratio > 1:
            new_width = self.target_size
            new_height = int(self.target_size / aspect_ratio)
        else:
            new_height = self.target_size
            new_width = int(self.target_size * aspect_ratio)
        
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        logging.debug(f"Resized mask from {width}x{height} to {new_width}x{new_height}")
        return resized_image
    
    def _create_mask_visualization(self, mask: np.ndarray) -> Image.Image:
        """Create visualization focusing only on mask quality."""
        
        # Ensure mask is single channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Convert to 3-channel for visualization
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        
        # Convert to PIL and resize for optimal VLM processing
        pil_image = Image.fromarray(mask_rgb)
        return self._resize_for_vlm(pil_image)
    
    def _get_artifact_prompt(self) -> str:
        """Get evaluation prompt focused specifically on mask artifacts."""
        return """You are an expert computer vision specialist evaluating ONLY mask quality for artifacts and defects.

TASK: Analyze the segmentation mask image (white = object, black = background).

Respond with ONLY this JSON format:
{
  "is_clean_mask": true/false,
  "confidence": 0.0-1.0
}

EVALUATION CRITERIA:

is_clean_mask: Is the mask clean without severe artifacts?

FAIL (mark FALSE) if you see:
- Severe fragmentation: Many scattered small white pieces (>10 disconnected blobs)
- Salt-and-pepper noise: Lots of tiny white dots scattered everywhere
- Swiss cheese effect: Large white regions with many black holes inside
- Extreme speckle patterns: White areas broken into hundreds of tiny pieces

PASS (mark TRUE) for:
- Solid white regions (1-5 main connected components)
- Minor edge roughness or small gaps
- A few small disconnected pieces (≤5 small extras)
- Generally cohesive white areas

FOCUS ONLY ON:
- Count disconnected white regions
- Look for excessive fragmentation
- Identify noise patterns

Be STRICT about obvious fragmentation but accept minor imperfections."""

    def _evaluate_artifacts(self, mask_image: Image.Image) -> Dict[str, Any]:
        """Evaluate mask artifacts using Gemma VLM."""
        
        if not self._model_loaded:
            self._load_model()
        
        system_prompt = self._get_artifact_prompt()
        
        messages = [
            {
                "role": "system", 
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": mask_image},
                    {"type": "text", "text": "Evaluate mask for artifacts:"}
                ]
            }
        ]

        model_inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            model_inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in model_inputs.items()}

        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs,
                max_new_tokens=80,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
            generation = generation[0][input_len:]

        response = self.processor.decode(generation, skip_special_tokens=True).strip()
        
        try:
            response = response.strip()
            if response.startswith('```json'):
                response = response.replace('```json', '').replace('```', '').strip()
            elif response.startswith('```'):
                response = response.replace('```', '').strip()
            
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                evaluation = json.loads(json_str)
            else:
                raise ValueError("No JSON object found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"Failed to parse artifact VLM response: {e}")
            response_lower = response.lower()
            is_clean = any(word in response_lower for word in ["true", "clean", "good", "solid"])
            
            evaluation = {
                "is_clean_mask": is_clean,
                "confidence": 0.4
            }
        
        return evaluation
    
    def filter(self, sample: Sample) -> FilterResult:
        """Evaluate mask artifacts."""
        mask = sample.load_mask(binary=False)
        
        mask_image = self._create_mask_visualization(mask)
        evaluation = self._evaluate_artifacts(mask_image)
        
        is_clean = evaluation.get("is_clean_mask", False)
        confidence = evaluation.get("confidence", 0.0)
        
        confidence_passed = confidence >= self.confidence_threshold
        final_passed = is_clean and confidence_passed
        
        if not final_passed:
            reasons = []
            if not is_clean:
                reasons.append("mask_artifacts_detected")
            if not confidence_passed:
                reasons.append(f"low_confidence={confidence:.3f}")
            reason = ", ".join(reasons)
        else:
            reason = None
        
        problem_desc = ""
        if not is_clean:
            problem_desc = "Severe Mask Artifacts"
        if not confidence_passed:
            problem_desc += f" | Low Conf ({confidence:.2f})" if problem_desc else f"Low Conf ({confidence:.2f})"
        
        return FilterResult(
            passed=final_passed,
            reason=reason,
            score=confidence,
            metadata={
                'artifact_evaluation': evaluation,
                'problem_description': problem_desc or f"Mask Clean [conf: {confidence:.2f}]"
            }
        )
