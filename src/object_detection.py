"""
Object Detection Module for Amazon Nova 2 Omni
"""

import json
import re
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from .common import call_nova_model

class ObjectDetector:
    """Object detection class"""
    
    def __init__(self):
        self.object_map = {
            "모든 객체": "objects and items",
            "사람": "people and persons", 
            "차량": "cars, trucks, motorcycles, bicycles, and vehicles",
            "동물": "animals, pets, dogs, cats, birds",
            "음식 & 음료": "food, drinks, fruits, vegetables, meals, beverages",
            "전자제품": "phones, computers, TVs, cameras, electronic devices",
            "가구": "chairs, tables, sofas, beds, furniture items",
            "스포츠": "balls, sports equipment, athletic gear",
            "도구": "tools, equipment, instruments, machinery",
            "식물": "plants, flowers, trees, vegetation"
        }
        
        # BGR colors for bounding boxes
        self.colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 165, 0)]
    
    def detect_objects(self, image, detection_type, custom_object=None, temperature=0.0, max_tokens=2000, top_p=1.0):
        """Detect objects in the image"""
        
        original_width, original_height = image.size
        
        buffer = BytesIO()
        if image.mode in ("RGBA", "P"):
            rgb_image = image.convert("RGB")
        else:
            rgb_image = image
        rgb_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        if detection_type == "사용자 정의" and custom_object:
            prompt = f"Detect all {custom_object} objects in this image. The image size is {original_width}x{original_height}. Return ONLY the bounding box coordinates in JSON format: [{{'bbox': [x1, y1, x2, y2], 'label': '{custom_object}'}}]. Use the exact image dimensions {original_width}x{original_height} for coordinates."
        else:
            object_desc = self.object_map.get(detection_type, "objects and items")
            prompt = f"Detect all {object_desc} in this image. The image size is {original_width}x{original_height}. Return ONLY the bounding box coordinates in JSON format: [{{'bbox': [x1, y1, x2, y2], 'label': 'object_name'}}]. Use the exact image dimensions {original_width}x{original_height} for coordinates."
        
        messages = [{
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": "png",
                        "source": {"bytes": image_bytes}
                    }
                },
                {"text": prompt}
            ]
        }]
        
        inference_config = {
            "temperature": temperature,
            "maxTokens": max_tokens,
            "topP": top_p
        }
        
        response = call_nova_model(messages, inference_config)
        
        if response and "output" in response:
            content = response["output"]["message"]["content"]
            
            detection_text = ""
            for item in content:
                if "text" in item:
                    detection_text = item["text"]
            
            # bounding box parsing and image drawing
            annotated_image, bbox_data = self._parse_and_draw_boxes(rgb_image, detection_text)
            
            return {
                "annotated_image": annotated_image,
                "detection_text": detection_text,
                "bbox_count": len(bbox_data) if bbox_data else 0,
                "original_size": (original_width, original_height)
            }
        
        return None
    
    def _parse_and_draw_boxes(self, rgb_image, detection_text):
        """Parse bounding boxes and draw them on the image"""
        
        try:
            # Extract and clean JSON coordinates
            json_match = re.search(r'\[.*\]', detection_text, re.DOTALL)
            if not json_match:
                return rgb_image, []
            
            json_str = json_match.group()
            json_str = json_str.replace("'bbox':", '"bbox":')
            json_str = json_str.replace("'label':", '"label":')
            json_str = json_str.replace("{'", '{"')
            json_str = json_str.replace("'}", '"}')
            json_str = json_str.replace("': ", '": ')
            json_str = json_str.replace(", '", ', "')
            
            try:
                bbox_data = json.loads(json_str)
            except json.JSONDecodeError:
                bbox_data = self._regex_parse_boxes(detection_text)
            
            if not bbox_data:
                return rgb_image, []
            
            # Convert PIL image to OpenCV format
            img_array = np.array(rgb_image)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            current_height, current_width = img_cv.shape[:2]
            
            # Find best scale for bounding boxes
            best_scale = self._find_best_scale(bbox_data, current_width, current_height)
            
            for i, obj in enumerate(bbox_data):
                bbox = obj['bbox']
                label = obj['label']
                color = self.colors[i % len(self.colors)]
                
                x1 = max(0, min(int(bbox[0] * best_scale[0]), current_width-1))
                y1 = max(0, min(int(bbox[1] * best_scale[1]), current_height-1))
                x2 = max(0, min(int(bbox[2] * best_scale[0]), current_width-1))
                y2 = max(0, min(int(bbox[3] * best_scale[1]), current_height-1))

                cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)
                
                label_text = f"{label} {i+1}"
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(img_cv, (x1, y1-30), (x1+label_size[0]+10, y1), color, -1)
                cv2.putText(img_cv, label_text, (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Convert back to PIL image
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            annotated_image = Image.fromarray(img_rgb)
            
            return annotated_image, bbox_data
            
        except Exception as e:
            return rgb_image, []
    
    def _regex_parse_boxes(self, detection_text):
        """정규식으로 바운딩 박스 파싱"""
        
        bbox_data = []
        object_pattern = r'\{[^}]+\}'
        objects = re.findall(object_pattern, detection_text)
        
        for obj_str in objects:
            # Extract bbox coordinates
            bbox_match = re.search(r'bbox[\'\"]*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', obj_str)
            # Extract label
            label_match = re.search(r'label[\'\"]*:\s*[\'\"](.*?)[\'"]', obj_str)
            
            if bbox_match and label_match:
                bbox_data.append({
                    'bbox': [int(bbox_match.group(1)), int(bbox_match.group(2)), 
                            int(bbox_match.group(3)), int(bbox_match.group(4))],
                    'label': label_match.group(1)
                })
        
        return bbox_data
    
    def _find_best_scale(self, bbox_data, current_width, current_height):
        """Find best scale for bounding boxes"""
        
        possible_scales = [
            (1.0, 1.0),  # original size
            (current_width/1024, current_height/1024),  # 1024 based
            (current_width/512, current_height/512),   # 512 based
            (current_width/800, current_height/600),   # 800x600 based
            (current_width/640, current_height/480),   # 640x480 based
            (current_width/1280, current_height/720),  # 720p based
            (current_width/1920, current_height/1080), # 1080p based
            (current_width/224, current_height/224),   # 224x224 based
        ]
        
        best_scale = (1.0, 1.0)
        best_score = 0
        
        for scale_x, scale_y in possible_scales:
            valid_boxes = 0
            for obj in bbox_data:
                bbox = obj['bbox']
                scaled_x1 = int(bbox[0] * scale_x)
                scaled_y1 = int(bbox[1] * scale_y)
                scaled_x2 = int(bbox[2] * scale_x)
                scaled_y2 = int(bbox[3] * scale_y)

                # Check if the scaled box is within image bounds and has valid size
                if (0 <= scaled_x1 < current_width and 
                    0 <= scaled_y1 < current_height and
                    0 <= scaled_x2 <= current_width and 
                    0 <= scaled_y2 <= current_height and
                    scaled_x2 > scaled_x1 and scaled_y2 > scaled_y1):
                    valid_boxes += 1
            
            if valid_boxes > best_score:
                best_score = valid_boxes
                best_scale = (scale_x, scale_y)
        
        return best_scale
