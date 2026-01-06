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
        
        # 이미지 전처리 (해상도 및 포맷 최적화)
        processed_image, processing_msg = self._preprocess_image(image)
        
        buffer = BytesIO()
        if processed_image.mode in ("RGBA", "P"):
            rgb_image = processed_image.convert("RGB")
        else:
            rgb_image = processed_image
        rgb_image.save(buffer, format="JPEG", quality=90)
        image_bytes = buffer.getvalue()
        
        if detection_type == "사용자 정의" and custom_object:
            prompt = f"Detect all {custom_object} objects in this image. Return ONLY a JSON array with bounding box coordinates: [{{'bbox': [x1, y1, x2, y2], 'label': '{custom_object}', 'confidence': 0.95}}]. Use normalized coordinates (0-1000 range)."
        else:
            object_desc = self.object_map.get(detection_type, "objects and items")
            prompt = f"Detect all {object_desc} in this image. Return ONLY a JSON array with bounding box coordinates: [{{'bbox': [x1, y1, x2, y2], 'label': 'object_name', 'confidence': 0.95}}]. Use normalized coordinates (0-1000 range)."
        
        messages = [{
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": "jpeg",
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
                "detection_json": bbox_data,
                "detection_text": detection_text,
                "bbox_count": len(bbox_data) if bbox_data else 0,
                "original_size": (original_width, original_height),
                "processing_message": processing_msg
            }
        
        return None
    
    def _preprocess_image(self, image):
        """이미지 전처리: 해상도 및 포맷 최적화"""
        processing_msg = ""
        processed_image = image
        
        width, height = image.size
        
        # 해상도 체크 및 리사이즈
        if width > 3000:
            # 비율 유지하면서 리사이즈
            ratio = 3000 / width
            new_width = 3000
            new_height = int(height * ratio)
            
            # OpenCV로 리사이즈
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_cv = img_array
            
            resized_cv = cv2.resize(img_cv, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            if len(img_array.shape) == 3:
                resized_rgb = cv2.cvtColor(resized_cv, cv2.COLOR_BGR2RGB)
            else:
                resized_rgb = resized_cv
            
            processed_image = Image.fromarray(resized_rgb)
            processing_msg += f"🔧 고해상도 이미지를 {width}x{height} → {new_width}x{new_height}로 리사이즈했습니다.\n"
        
        # PNG를 JPEG로 변환 (투명도 제거)
        if image.format == 'PNG' or processed_image.mode in ('RGBA', 'P'):
            if processed_image.mode in ('RGBA', 'P'):
                processed_image = processed_image.convert('RGB')
            processing_msg += "🔧 PNG 이미지를 JPEG로 변환하여 처리 속도를 최적화했습니다.\n"
        
        return processed_image, processing_msg
    
    def _parse_and_draw_boxes(self, rgb_image, detection_text):
        """Parse bounding boxes and draw them on the image"""
        
        try:
            # 여러 방법으로 JSON 파싱 시도
            bbox_data = self._extract_json_data(detection_text)
            
            if not bbox_data:
                return rgb_image, []
            
            # Convert PIL image to OpenCV format
            img_array = np.array(rgb_image)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            current_height, current_width = img_cv.shape[:2]
            
            for i, obj in enumerate(bbox_data):
                bbox = obj['bbox']
                label = obj['label']
                color = self.colors[i % len(self.colors)]
                
                # Remap normalized coordinates (0-1000) to image dimensions
                remapped_bbox = self._remap_bbox_to_image(bbox, current_width, current_height)
                
                x1 = max(0, min(int(remapped_bbox[0]), current_width-1))
                y1 = max(0, min(int(remapped_bbox[1]), current_height-1))
                x2 = max(0, min(int(remapped_bbox[2]), current_width-1))
                y2 = max(0, min(int(remapped_bbox[3]), current_height-1))

                cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)
                
                label_text = f"{label} {i+1}"
                (label_width, label_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_y = max(label_height + 5, y1)
                cv2.rectangle(img_cv, (x1, label_y - label_height - 5), (x1 + label_width + 10, label_y), color, -1)
                cv2.putText(img_cv, label_text, (x1 + 5, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Convert back to PIL image
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            annotated_image = Image.fromarray(img_rgb)
            
            return annotated_image, bbox_data
            
        except Exception as e:
            return rgb_image, []
    
    def _extract_json_data(self, detection_text):
        """다양한 방법으로 JSON 데이터 추출"""
        
        # 방법 1: 표준 JSON 배열 추출
        json_match = re.search(r'\[.*\]', detection_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            
            # JSON 문자열 정리
            json_str = self._clean_json_string(json_str)
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # 방법 2: 정규식으로 파싱
        return self._regex_parse_boxes(detection_text)
    
    def _clean_json_string(self, json_str):
        """JSON 문자열 정리"""
        # 작은따옴표를 큰따옴표로 변경
        json_str = json_str.replace("'bbox':", '"bbox":')
        json_str = json_str.replace("'label':", '"label":')
        json_str = json_str.replace("'confidence':", '"confidence":')
        json_str = json_str.replace("{'", '{"')
        json_str = json_str.replace("'}", '"}')
        json_str = json_str.replace("': ", '": ')
        json_str = json_str.replace(", '", ', "')
        
        # confidence 값의 문자열 따옴표 제거
        json_str = re.sub(r'"confidence":\s*"([0-9.]+)"', r'"confidence": \1', json_str)
        
        return json_str
    
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
    
    def _remap_bbox_to_image(self, bounding_box, image_width, image_height):
        """Remap normalized coordinates (0-1000) to image dimensions"""
        return [
            bounding_box[0] * image_width / 1000,
            bounding_box[1] * image_height / 1000,
            bounding_box[2] * image_width / 1000,
            bounding_box[3] * image_height / 1000,
        ]
