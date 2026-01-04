"""
Image Generation Classes and Functions
"""
import base64
from io import BytesIO
from PIL import Image
from .common import get_bedrock_runtime, get_current_model_id, translate_to_english, detect_non_english, call_nova_model

class ImageGenerator:
    """Image generation using Nova 2 Omni"""
    
    def __init__(self):
        self.bedrock = get_bedrock_runtime()
    
    def generate_image(self, prompt: str, visual_style: str = "사실적 이미지 (기본)", 
                      aspect_ratio: str = "16:9 (기본)", temperature: float = 0.7, 
                      max_tokens: int = 4000, top_p: float = 1.0, reasoning: bool = False) -> dict:
        """Generate image with given parameters"""
        
        # Translate prompt if needed
        if detect_non_english(prompt):
            english_prompt = translate_to_english(prompt)
        else:
            english_prompt = prompt
        
        # Apply visual style
        styled_prompt = self._apply_visual_style(english_prompt, visual_style)
        
        # Apply aspect ratio
        final_prompt = self._apply_aspect_ratio(styled_prompt, aspect_ratio)
        
        # System prompt
        system_prompt = self._get_system_prompt(visual_style)
        
        # Message construction
        messages = [{
            "role": "user",
            "content": [{"text": f"Create an image: {final_prompt}"}]
        }]
        
        # Configure inference settings
        inference_config = {
            "temperature": temperature,
            "maxTokens": max_tokens,
            "topP": top_p
        }
        
        request_config = {}
        if system_prompt:
            request_config["system"] = [{"text": system_prompt}]
        
        if reasoning:
            request_config["additionalModelRequestFields"] = {
                "reasoningConfig": {
                    "type": "enabled",
                    "maxReasoningEffort": "medium"
                }
            }
        
        # Call API
        response = call_nova_model(messages, inference_config, request_config)
        
        # Extract image from response
        if response and "output" in response:
            content = response["output"]["message"]["content"]
            for item in content:
                if "image" in item:
                    image_source = item["image"]["source"]
                    if "bytes" in image_source:
                        if isinstance(image_source["bytes"], str):
                            image_data = base64.b64decode(image_source["bytes"])
                        else:
                            image_data = image_source["bytes"]
                        
                        generated_image = Image.open(BytesIO(image_data))
                        return {
                            "success": True,
                            "image": generated_image,
                            "prompt": final_prompt,
                            "style": visual_style
                        }
        
        return {"success": False, "error": "이미지를 생성할 수 없습니다."}
    
    def _apply_visual_style(self, prompt: str, style: str) -> str:
        """Apply visual style to prompt"""
        style_prompts = {
            "사실적 이미지 (기본)": prompt,
            "스토리북 일러스트": f"{prompt}. Whimsical storybook illustration style.",
            "애니메이션 (2D - 일본풍)": f"{prompt}. Japanese anime style, 2D animation with vibrant colors and expressive characters.",
            "애니메이션 (2D - 서양풍)": f"{prompt}. Western 2D animation style, Disney-like cartoon animation.",
            "애니메이션 (3D)": f"{prompt}. 3D animated movie style, Pixar-like rendering.",
            "디지털 아트": f"{prompt}. Digital art style, vibrant colors and sharp details.",
            "수채화": f"{prompt}. Watercolor painting style, soft and flowing.",
            "유화": f"{prompt}. Oil painting style, rich textures and brushstrokes.",
            "만화/카툰": f"{prompt}. Cartoon comic book style, bold lines and bright colors.",
            "미니멀": f"{prompt}. Minimalist style, clean and simple composition.",
            "빈티지": f"{prompt}. Vintage retro style, aged and nostalgic feel."
        }
        return style_prompts.get(style, prompt)
    
    def _apply_aspect_ratio(self, prompt: str, aspect_ratio: str) -> str:
        """Apply aspect ratio to prompt"""
        aspect_ratio_prompts = {
            "16:9 (기본)": "in 16:9 aspect ratio",
            "2:1 (와이드)": "in 2:1 wide panoramic format", 
            "3:2 (가로)": "in 3:2 landscape format",
            "4:3 (가로)": "in 4:3 horizontal format",
            "1:1 (정사각형)": "in 1:1 square format",
            "1:2 (세로)": "in 1:2 tall vertical format",
            "9:16 (세로)": "in 9:16 mobile vertical format", 
            "2:3 (세로)": "in 2:3 portrait format",
            "3:4 (세로)": "in 3:4 vertical format"
        }
        
        aspect_prompt = aspect_ratio_prompts.get(aspect_ratio, "")
        return f"{prompt} {aspect_prompt}" if aspect_prompt else prompt
    
    def _get_system_prompt(self, style: str) -> str:
        """Get system prompt for visual style"""
        system_prompts = {
            "사실적 이미지 (기본)": "You are a professional photographer creating realistic, high-quality photographic images.",
            "스토리북 일러스트": "You are a children's book illustrator creating whimsical, charming storybook illustrations.",
            "애니메이션 (2D - 일본풍)": "You are a Japanese anime artist creating vibrant 2D anime-style illustrations with expressive characters.",
            "애니메이션 (2D - 서양풍)": "You are a Western animation artist creating Disney-style 2D cartoon animations.",
            "애니메이션 (3D)": "You are a 3D animation artist creating Pixar-style animated movie scenes.",
            "디지털 아트": "You are a digital artist creating vibrant, modern digital artwork with sharp details.",
            "수채화": "You are a watercolor artist creating soft, flowing paintings with gentle color blends.",
            "유화": "You are an oil painter creating rich, textured paintings with visible brushstrokes.",
            "만화/카툰": "You are a comic book artist creating bold, colorful cartoon-style illustrations.",
            "미니멀": "You are a minimalist artist creating clean, simple compositions with elegant restraint.",
            "빈티지": "You are a vintage artist creating nostalgic, retro-style artwork with aged aesthetics."
        }
        return system_prompts.get(style, "You are a professional image generator.")
