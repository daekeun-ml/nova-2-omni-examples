"""
Common utilities and configurations for Nova 2 Omni
"""
import boto3
from botocore.config import Config
from io import BytesIO
from PIL import Image

# Common constants (기본값)
DEFAULT_MODEL_ID = "us.amazon.nova-2-omni-v1:0"
DEFAULT_REGION_ID = "us-west-2"

def get_current_model_id():
    """현재 설정된 모델 ID 반환"""
    import streamlit as st
    return getattr(st.session_state, 'model_id', DEFAULT_MODEL_ID)

def get_current_region_id():
    """현재 설정된 리전 ID 반환"""
    import streamlit as st
    return getattr(st.session_state, 'region_id', DEFAULT_REGION_ID)

def parse_json_from_text(text):
    """텍스트에서 JSON을 추출하고 파싱하는 공통 함수"""
    import json
    
    # Extract JSON from markdown code blocks if present
    if '```json' in text:
        json_str = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        json_str = text.split('```')[1].split('```')[0].strip()
    else:
        # JSON 객체 또는 배열 찾기
        import re
        # 객체 형태 먼저 시도
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not json_match:
            # 배열 형태 시도
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group()
        else:
            json_str = text
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # JSON 문자열 정리 시도 (따옴표 수정)
        json_str = json_str.replace("'bbox':", '"bbox":')
        json_str = json_str.replace("'label':", '"label":')
        json_str = json_str.replace("{'", '{"')
        json_str = json_str.replace("'}", '"}')
        json_str = json_str.replace("': ", '": ')
        json_str = json_str.replace(", '", ', "')
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None


def get_bedrock_runtime(region_id=None):
    """Returns a properly configured Bedrock Runtime client."""
    if region_id is None:
        region_id = get_current_region_id()
    
    config = Config(read_timeout=2 * 60)
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=region_id,
        config=config,
    )

def convert_image_to_bytes(image):
    """Convert PIL Image to bytes"""
    buffer = BytesIO()
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    return buffer.getvalue(), "png"

def load_image_as_bytes(uploaded_file):
    """Nova 2 Omni 이미지 편집 요구사항에 맞게 이미지 처리"""
    # 이미지 로드
    image = Image.open(uploaded_file)
    
    # 현재 픽셀 수 계산
    current_pixels = image.width * image.height
    max_pixels = 1048576  # 1 megapixel for editing
    
    # 크기 조정이 필요한 경우
    if current_pixels > max_pixels:
        ratio = (max_pixels / current_pixels) ** 0.5
        new_width = int(image.width * ratio)
        new_height = int(image.height * ratio)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # RGB로 변환 (RGBA나 P 모드 처리)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    
    # PNG로 저장
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    
    return buffer.getvalue(), "png"

def call_nova_model(messages, inference_config=None, request_config=None):
    """Nova 모델 호출"""
    import time
    import streamlit as st
    
    # 현재 설정값 가져오기
    model_id = get_current_model_id()
    region_id = get_current_region_id()
    
    bedrock = get_bedrock_runtime(region_id)
    
    request = {
        "modelId": model_id,
        "messages": messages
    }
    
    if inference_config:
        request["inferenceConfig"] = inference_config
    
    if request_config:
        request.update(request_config)
    
    try:
        st.write(f"🔄 API 요청 시작: {time.strftime('%H:%M:%S')}")
        st.write(f"📊 메시지 크기: {len(str(messages))} 문자")
        if inference_config:
            st.write(f"⚙️ 설정: Temperature={inference_config.get('temperature', 'N/A')}, MaxTokens={inference_config.get('maxTokens', 'N/A')}, TopP={inference_config.get('topP', 'N/A')}")
        
        start_time = time.time()
        response = bedrock.converse(**request)
        end_time = time.time()
        
        st.write(f"✅ API 응답 완료: {time.strftime('%H:%M:%S')} (소요시간: {end_time - start_time:.1f}초)")
        return response
        
    except Exception as e:
        st.error(f"API 호출 오류: {e}")
        st.write(f"오류 타입: {type(e).__name__}")
        return None

def detect_non_english(text):
    """텍스트에 비영어 문자가 포함되어 있는지 확인"""
    import re
    # 한국어, 중국어, 일본어 패턴
    non_english_pattern = re.compile(r'[가-힣一-龯ひらがなカタカナ]')
    return bool(non_english_pattern.search(text))

def translate_text(text, target_language="English"):
    """텍스트를 지정된 언어로 번역"""
    bedrock = get_bedrock_runtime()
    
    if target_language == "English":
        prompt = f"""Translate the following text to English. Make sure to translate ALL words including proper nouns and specific terms. For example:
- 호랑이 → tiger
- 고양이 → cat  
- 강아지 → dog
- 자동차 → car

Text to translate: {text}

Provide ONLY the complete English translation:"""
    else:
        prompt = f"""Translate the following text to {target_language} using consistent formal/polite language throughout. Maintain a respectful and consistent tone without mixing formal and informal expressions. Use natural, fluent expressions appropriate for the context and use appropriate counters and units.

Text to translate: {text}

Provide ONLY the complete {target_language} translation using consistent formal language:"""
    
    translate_messages = [{
        "role": "user",
        "content": [{"text": prompt}]
    }]
    
    try:
        response = bedrock.converse(
            modelId=get_current_model_id(),
            messages=translate_messages,
            inferenceConfig={"temperature": 0.1, "maxTokens": 1024}
        )
        
        if response and "output" in response:
            translated = response["output"]["message"]["content"][0]["text"]
            # 기본적인 정리만 (앞뒤 공백, 따옴표 제거)
            translated = translated.strip().strip('"\'')
            
            # "Translation:" 같은 접두사가 있으면 제거
            prefixes = ['translation:', f'{target_language.lower()}:', 'result:', '번역:', '翻译:', '翻訳:']
            for prefix in prefixes:
                if translated.lower().startswith(prefix):
                    translated = translated.split(':', 1)[1].strip()
                    break
            
            return translated
    except:
        pass
    
    return text  # 번역 실패 시 원본 반환

def translate_to_english(text):
    """비영어 텍스트를 영어로 번역 (기존 호환성 유지)"""
    return translate_text(text, "English")

def extract_video_frames(video_bytes, timestamps):
    """비디오에서 특정 타임스탬프의 프레임들을 추출"""
    import tempfile
    import os
    import cv2
    
    # 임시 비디오 파일 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_bytes)
        tmp_video_path = tmp_file.name
    
    try:
        cap = cv2.VideoCapture(tmp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames = []
        for timestamp in timestamps:
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                # BGR to RGB 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            else:
                frames.append(None)
        
        cap.release()
        return frames
        
    finally:
        # 임시 파일 정리
        if os.path.exists(tmp_video_path):
            os.unlink(tmp_video_path)

def format_stt_result(text):
    """STT 결과 가독성 향상을 위한 포맷팅"""
    if not text:
        return text
    
    # 문장의 첫 글자와 마침표 후 첫 글자를 대문자로 변환
    sentences = text.split('. ')
    formatted_sentences = []
    
    for sentence in sentences:
        if sentence:
            # 첫 글자 대문자화
            formatted_sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
            formatted_sentences.append(formatted_sentence)
    
    return '. '.join(formatted_sentences)

def translate_to_user_language(text):
    """영어 텍스트를 사용자 언어로 번역 (한국어 기본)"""
    return translate_text(text, "Korean")
