"""
Video Understanding Classes and Functions
"""
import json
import zipfile
from io import BytesIO
from PIL import Image
from botocore.exceptions import ClientError
from .common import get_bedrock_runtime, get_current_model_id, extract_video_frames, parse_json_from_text, translate_text

class VideoAnalyzer:
    """Video analysis using Nova 2 Omni"""
    
    def __init__(self):
        self.bedrock = get_bedrock_runtime()
    
    def analyze_video(self, video_bytes: bytes, video_format: str, prompt: str, 
                     temperature: float = 0.0, top_p: float = 1.0, max_tokens: int = 10000) -> str:
        """Analyze video with given prompt"""
        request = {
            "modelId": get_current_model_id(),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"video": {"format": video_format, "source": {"bytes": video_bytes}}},
                        {"text": prompt},
                    ],
                }
            ],
            "inferenceConfig": {"temperature": temperature, "topP": top_p, "maxTokens": max_tokens},
        }
        
        try:
            response = self.bedrock.converse(**request)
            return response
                
        except ClientError as err:
            return {"error": f"Error occurred: {err}"}
    
    def process_highlights(self, result_text: str, video_bytes: bytes) -> dict:
        """Process highlight extraction results and generate thumbnails using extract_video_frames()"""
        try:
            digest_data = parse_json_from_text(result_text)
            if not digest_data or "highlights" not in digest_data:
                return {"success": False, "error": "하이라이트 데이터를 파싱할 수 없습니다."}
            
            timestamps = []
            for h in digest_data["highlights"]:
                timestamp = h.get("timestamp", 0)
                # timestamp가 문자열인 경우 float로 변환
                if isinstance(timestamp, str):
                    try:
                        # Convert "MM:SS" format to seconds
                        if ":" in timestamp:
                            parts = timestamp.split(":")
                            timestamp = float(parts[0]) * 60 + float(parts[1])
                        else:
                            timestamp = float(timestamp)
                    except:
                        timestamp = 0.0
                timestamps.append(float(timestamp))
            
            frames = extract_video_frames(video_bytes, timestamps)
            
            highlight_frames = []
            for highlight, frame in zip(digest_data["highlights"], frames):
                if frame is not None:
                    timestamp = highlight.get("timestamp", 0)
                    if isinstance(timestamp, str):
                        try:
                            if ":" in timestamp:
                                parts = timestamp.split(":")
                                timestamp = float(parts[0]) * 60 + float(parts[1])
                            else:
                                timestamp = float(timestamp)
                        except:
                            timestamp = 0.0
                    
                    highlight_frames.append({
                        "frame": frame,
                        "timestamp": float(timestamp),
                        "keywords": highlight.get("keywords", []),
                        "impact": highlight.get("impact", highlight.get("importance", "N/A")),
                        "description": translate_text(highlight.get("description", ""), "Korean")
                    })
            
            if not highlight_frames:
                return {"success": False, "error": "하이라이트 프레임을 추출할 수 없습니다."}
            
            # Make a zip file of highlight images
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                for idx, highlight in enumerate(highlight_frames):
                    pil_image = Image.fromarray(highlight["frame"])
                    img_buffer = BytesIO()
                    pil_image.save(img_buffer, format="PNG")
                    
                    filename = f"highlight_{idx+1}_{highlight['timestamp']:.1f}s_impact{highlight['impact']}.png"
                    zip_file.writestr(filename, img_buffer.getvalue())
            
            return {
                "success": True,
                "highlight_frames": highlight_frames,
                "zip_data": zip_buffer.getvalue()
            }
            
        except Exception as e:
            return {"success": False, "error": f"하이라이트 처리 중 오류: {str(e)}"}

# Predefined analysis types with prompts (from Jupyter notebook)
VIDEO_ANALYSIS_PROMPTS = {
    "요약": "Can you create an executive summary of this video's content?",
    
    "하이라이트 추출": """Extract high-impact highlight moments from this video and provide:
1. A digest summary of the most important moments
2. Key highlight timestamps with descriptions
3. Generate representative images for each highlight moment
4. Overall summary of the video content

Format the response as JSON:
{
  "digest": "Brief summary of key moments",
  "highlights": [
    {
      "timestamp": "MM:SS",
      "description": "What happens at this moment",
      "importance": "high/medium/low"
    }
  ],
  "summary": "Overall video summary",
  "key_topics": ["topic1", "topic2", "topic3"]
}""",
    
    "시각적 설명": """Provide a rich visual description of this video. Focus on:
- Camera angles and framing (top-down, close-up, etc.)
- Color palette and lighting
- Visual composition and layout
- Movement and transitions
- Text overlays and their styling
- Overall aesthetic and production quality

Describe what makes this video visually engaging.""",
    
    "이벤트 타임스탬프": lambda event_query: f"Please localize the moment that the event '{event_query}' happens in the video. Answer with the starting and ending time of the event in seconds. e.g. [[72, 82]]. If the event happen multiple times, list all of them. e.g. [[40, 50], [72, 82]]",
    
    "비디오 세그멘테이션": "Segment a video into different scenes and generate caption per scene. The output should be in the format: [STARTING TIME-ENDING TIMESTAMP] CAPTION. Timestamp in MM:SS format",
    
    "비디오 분류": """What is the most appropriate category for this video? Select your answer from the options provided:
Cooking Tutorial
Home Repair
Makeup Tutorial
Sports
Education
Entertainment
News
Other

Provide the category and a brief explanation."""
}

def parse_timestamps(result: str) -> list:
    """Extract timestamps from event timestamp result"""
    import re
    # Find all timestamp patterns like [[19.0, 20.0]] or [[40, 50], [72, 82]]
    pattern = r'\[\[([0-9.]+),\s*([0-9.]+)\]\]'
    matches = re.findall(pattern, result)
    
    timestamps = []
    for match in matches:
        start_time = float(match[0])
        end_time = float(match[1])
        timestamps.append((start_time, end_time))
    
    return timestamps

def format_video_result(result: str, analysis_type: str) -> str:
    """Format video analysis result for better readability"""
    
    if analysis_type == "비디오 세그멘테이션":
        # Format video segmentation with markdown
        lines = result.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line and '[' in line and ']' in line:
                # Extract timestamp and description
                parts = line.split('] ', 1)
                if len(parts) == 2:
                    timestamp = parts[0] + ']'
                    description = parts[1]
                    formatted_lines.append(f"**{timestamp}**\n\n{description}\n")
                else:
                    formatted_lines.append(line + '\n')
            elif line:
                formatted_lines.append(line + '\n')
        
        return '\n'.join(formatted_lines)
    
    elif analysis_type == "비디오 분류":
        # Format classification with emphasis
        lines = result.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if any(category in line for category in ['Cooking Tutorial', 'Home Repair', 'Makeup Tutorial', 'Sports', 'Education', 'Entertainment', 'News', 'Other']):
                formatted_lines.append(f"**분류 결과: {line}**\n")
            elif line:
                formatted_lines.append(line + '\n')
        
        return '\n'.join(formatted_lines)
    
    else:
        # Default formatting for other types
        return result

def get_video_format(filename: str) -> str:
    """Get video format from filename"""
    extension = filename.lower().split('.')[-1]
    format_map = {
        'mp4': 'mp4',
        'mov': 'mov', 
        'avi': 'avi',
        'mkv': 'mkv',
        'webm': 'webm'
    }
    return format_map.get(extension, 'mp4')
