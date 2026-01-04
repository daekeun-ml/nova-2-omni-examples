"""
Image Editing Classes and Functions
"""
import base64
from io import BytesIO
from PIL import Image
from .common import get_bedrock_runtime, get_current_model_id, translate_to_english, detect_non_english, call_nova_model, load_image_as_bytes

class ImageEditor:
    """Image editing using Nova 2 Omni"""
    
    def __init__(self):
        self.bedrock = get_bedrock_runtime()
    
    def edit_image(self, uploaded_file, edit_type: str, temperature: float = 0.0, 
                   max_tokens: int = 4000, top_p: float = 1.0, **kwargs) -> dict:
        """Edit image with given parameters"""
        
        # Generate edit prompt
        edit_prompt = self._generate_edit_prompt(edit_type, **kwargs)
        
        # Translate prompt if needed
        if detect_non_english(edit_prompt):
            english_edit_prompt = translate_to_english(edit_prompt)
        else:
            english_edit_prompt = edit_prompt
        
        # Prepare image data
        uploaded_file.seek(0)
        image_bytes, image_format = load_image_as_bytes(uploaded_file)
        
        # Message construction
        messages = [{
            "role": "user",
            "content": [
                {"image": {"format": image_format, "source": {"bytes": image_bytes}}},
                {"text": f"Edit this image: {english_edit_prompt}. Please generate a new edited image, not text instructions."}
            ]
        }]
        
        system_prompt = """You are a professional photo editor specializing in precise background replacement. When editing images:
1. For background changes: ONLY modify the background - never change people, faces, clothing, or main objects
2. Think of it as digital compositing - cut out subjects and place on new background
3. Preserve ALL original subject details: faces, expressions, clothing, poses, objects
4. For other edits: Make clear, visible changes as requested
5. Always generate a new edited image with the requested modifications
6. Maintain photorealistic quality and proper lighting"""
        
        # Configure inference settings
        inference_config = {
            "temperature": temperature,
            "maxTokens": max_tokens,
            "topP": top_p
        }
        
        request_config = {
            "system": [{"text": system_prompt}]
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
                        
                        edited_image = Image.open(BytesIO(image_data))
                        return {
                            "success": True,
                            "image": edited_image,
                            "prompt": english_edit_prompt
                        }
        
        return {"success": False, "error": "편집된 이미지를 생성할 수 없습니다."}
    
    def _generate_edit_prompt(self, edit_type: str, **kwargs) -> str:
        """Generate edit prompt based on type and parameters"""
        
        if edit_type == "사용자 정의":
            return kwargs.get("edit_prompt", "")
        
        elif edit_type == "텍스트 추가":
            text_content = kwargs.get("text_content", "Welcome")
            text_position = kwargs.get("text_position", "유리창 중앙에")
            text_style = kwargs.get("text_style", "유리창 글씨")
            
            style_prompts = {
                "유리창 글씨": f"The words '{text_content}' can be seen on the glass window, written in an elegant calligraphy font that looks naturally etched or painted on the glass surface",
                "간판": f"Add a realistic shop sign with '{text_content}' written in professional signage lettering, properly mounted and lit",
                "벽면 페인팅": f"The text '{text_content}' is painted directly on the wall surface with realistic paint texture and proper shadowing",
                "네온사인": f"Create a glowing neon sign displaying '{text_content}' with realistic neon tube lighting effects",
                "조각/새김": f"The text '{text_content}' appears carved or engraved into the surface with realistic depth and shadow effects"
            }
            
            return f"Transform this image to show {style_prompts[text_style]} {text_position}. Ensure the text integrates naturally with the scene's lighting, perspective, and materials."
        
        elif edit_type == "사물/인물 추가":
            object_to_add = kwargs.get("object_to_add", "고양이")
            add_position = kwargs.get("add_position", "왼쪽")
            integration_style = kwargs.get("integration_style", "자연스럽게")
            
            position_map = {"왼쪽": "left side", "오른쪽": "right side", "중앙": "center", "배경": "background", "전경": "foreground"}
            style_map = {"자연스럽게": "naturally", "사실적으로": "realistically", "조화롭게": "harmoniously"}
            
            return f"Add a {object_to_add} to the {position_map[add_position]} of this image {style_map[integration_style]}. Ensure proper lighting, shadows, and perspective to make it look like it belongs in the original scene."
        
        elif edit_type == "사물/인물 제거":
            object_to_remove = kwargs.get("object_to_remove", "반찬통")
            return f"Remove the {object_to_remove} from this image"
        
        elif edit_type == "배경 변경":
            new_background = kwargs.get("new_background", "바다")
            transition_style = kwargs.get("transition_style", "자연스럽게")
            
            style_map = {
                "자연스럽게": f"This is a background replacement task. You must ONLY change the background to {new_background}. The people, faces, clothing, poses, and all foreground objects must remain IDENTICAL to the original image. Think of this as cutting out the subjects and pasting them onto a new {new_background}",
                "완전히 교체": f"Replace ONLY the background with {new_background}. All people and main objects must look EXACTLY the same as in the original - same faces, same clothes, same poses, same everything. Only the background scenery changes",
                "부분적으로": f"Partially blend {new_background} into the background while keeping ALL people and main objects completely unchanged from the original image"
            }
            
            return f"{style_map[transition_style]}. IMPORTANT: This is NOT a style transfer or artistic transformation - it's a precise background replacement. Keep all subjects identical to the original."
        
        elif edit_type == "색상 변경":
            target_object = kwargs.get("target_object", "자전거")
            new_color = kwargs.get("new_color", "빨간색")
            return f"Change the color of the {target_object} to {new_color}"
        
        elif edit_type == "스타일 변경":
            new_style = kwargs.get("new_style", "애니메이션 (2D - 일본풍)")
            return f"Transform this image completely into {new_style} style. Make dramatic visual changes to convert the entire image style, colors, textures, and artistic rendering to match {new_style} aesthetic. The result should look distinctly different from the original with clear {new_style} characteristics."
        
        return ""
