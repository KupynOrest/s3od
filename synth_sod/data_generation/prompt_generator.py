import random
from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


class PromptEnhancer:
    def __init__(self):
        self.color_balance = [
            "natural colors", "vibrant colors", "true colors", "balanced color temperature",
            "daylight color balance", "neutral white balance", "clear colors"
        ]
        
        self.clarity_terms = [
            "sharp details", "clear image", "no filter", "natural lighting", 
            "unprocessed", "raw photo style", "clean image"
        ]
        
        self.lighting_variety = [
            "bright daylight", "cool lighting", "blue hour lighting", "overcast lighting",
            "studio lighting", "fluorescent lighting", "LED lighting"
        ]
        
        self.complexity_terms = [
            "sharp focus throughout", "everything in focus", "deep depth of field", "no bokeh",
            "complex background", "detailed background", "cluttered scene", "busy environment", 
            "multiple objects", "overlapping elements", "textured surfaces"
        ]

    def __call__(self, prompt: str) -> str:
        enhanced_prompt = prompt.strip()
        
        # 30% chance to add color balance correction
        if random.random() < 0.3:
            color_term = random.choice(self.color_balance)
            enhanced_prompt = f"{enhanced_prompt}, {color_term}"
        
        # 25% chance to add clarity enhancement
        if random.random() < 0.25:
            clarity_term = random.choice(self.clarity_terms)
            enhanced_prompt = f"{enhanced_prompt}, {clarity_term}"
        
        # 20% chance to add specific lighting to counter brownish bias
        if random.random() < 0.2:
            lighting_term = random.choice(self.lighting_variety)
            enhanced_prompt = f"{enhanced_prompt}, {lighting_term}"
        
        # 25% chance to add complexity terms to counter bokeh and increase difficulty
        if random.random() < 0.25:
            complexity_term = random.choice(self.complexity_terms)
            enhanced_prompt = f"{enhanced_prompt}, {complexity_term}"
        
        return enhanced_prompt


class ImagePromptGenerator:
    def __init__(self, model="gpt-4o"):
        self.llm = ChatOpenAI(model=model)

        self.prompt = PromptTemplate(
            input_variables=["num_prompts", "main_class"],
            template="""Generate exactly {num_prompts} diverse, photorealistic prompts for {main_class} images for salient object detection. Create natural scenes with varying complexity levels.

                    Requirements:
                    - Photorealistic scenes only - no artistic or cartoon styles
                    - Main object clearly visible and identifiable
                    - Sharp focus throughout the scene
                    - Natural lighting and environments

                    Vary these aspects across prompts:
                    
                    Object variations: Different sizes, positions, quantities (1-3 objects), conditions (new/worn), and orientations
                    
                    Scene complexity: Mix simple and complex backgrounds - from clean settings to cluttered environments with multiple objects and busy textures
                    
                    Lighting conditions: Natural daylight, golden hour, overcast skies, indoor lighting, shadows and highlights
                    
                    Environments: Indoor/outdoor settings, natural habitats, functional contexts (object being used), storage areas, seasonal variations
                    
                    Visual challenges: Include some scenes with partial occlusion, similar colors, reflective surfaces, overlapping objects, or camouflage effects when natural
                    
                    Perspectives: Close-ups, medium shots, wide views, and varied camera angles (above, below, side views)
                    
                    Context diversity: Objects in use, at rest, in groups, in natural habitats, different weather conditions, times of day

                    Balance: Create a mix of challenging segmentation scenarios.

                    Return exactly {num_prompts} prompts as Python list:
                    ["A scene description...", ...]

                    Important: Maximize diversity - avoid repetitive scenarios or settings."""
        )

    def generate_prompts(self, main_class: str, num_prompts: int = 100) -> List[str]:
        """Generate diverse image prompts for the given class"""

        formatted_prompt = self.prompt.format(
            num_prompts=num_prompts,
            main_class=main_class
        )

        messages = [
            SystemMessage(content="You are a helpful assistant that generates image prompts for salient object detection synthetic data generation pipeline."),
            HumanMessage(content=formatted_prompt)
        ]

        response = self.llm.invoke(messages)

        try:
            start_idx = response.content.find('[')
            end_idx = response.content.rfind(']') + 1
            return eval(response.content[start_idx:end_idx])
        except Exception as e:
            print(f"Error parsing prompts: {e}")
            return []
