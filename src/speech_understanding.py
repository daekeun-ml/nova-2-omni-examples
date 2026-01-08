"""
Speech Understanding Classes and Functions
"""
from .common import get_bedrock_runtime, get_current_model_id

class SpeechAnalyzer:
    """Speech understanding using Nova 2 Omni"""
    
    def __init__(self):
        self.bedrock = get_bedrock_runtime()
    
    def analyze_audio(self, audio_bytes: bytes, audio_format: str, analysis_type: str, 
                     question: str = None, temperature: float = 0.0, max_tokens: int = 4000, top_p: float = 1.0) -> str:
        """Analyze audio content"""
        
        # Analysis type prompts
        prompts = {
            "transcription": "Transcribe the audio.",
            "diarization": """For each speaker turn segment, transcribe, assign a speaker label, start and end timestamps. 
You must follow the exact XML format shown in the example below: 
'<segment><transcription speaker="speaker_id" start="start_time" end="end_time">transcription_text</transcription></segment>'""",
            "summary": """Create a comprehensive summary of this audio content in well-structured markdown format:

## 📋 주요 내용 요약

### 🎯 핵심 주제
- [주요 주제들을 bullet point로 나열]

### 📝 상세 내용
- [중요한 내용들을 체계적으로 정리]

### 💡 주요 포인트
- [핵심 포인트들을 명확하게 정리]

### 📊 결론 및 시사점
- [결론과 중요한 시사점들을 정리]

Please format the response in clean, readable Korean markdown with proper headers and bullet points.""",
            "sentiment": "Analyze the sentiment and emotional tone of the speakers in this audio.",
            "key_points": "Extract the key points, important topics, and main takeaways from this audio.",
            "call_analytics": """Analyze the call and return JSON:
{
  "call_summary": "Summarize the call",
  "customer_intent": "What the customer wanted",
  "resolution_status": "resolved/pending/escalated",
  "key_topics": ["topic1", "topic2"],
  "action_items": [
    {"task": "description", "owner": "agent/customer", "priority": "high/medium/low"}
  ],
  "sentiment_analysis": {
    "overall": "positive/neutral/negative"
  },
  "follow_up_required": true/false
}""",
            "qa": question if question else "Answer questions about this audio content.",
            "translation": "Transcribe and translate this audio to English if it's in another language, or provide a summary if it's already in English."
        }
        
        if analysis_type == "qa" and question:
            # Q&A의 경우 한국어 질문을 영어로 번역
            from .common import detect_non_english, translate_to_english
            if detect_non_english(question):
                english_question = translate_to_english(question)
                prompt = f"Answer this question about the audio: {english_question}"
            else:
                prompt = f"Answer this question about the audio: {question}"
        else:
            prompt = prompts.get(analysis_type, prompts["transcription"])
        
        messages = [{
            "role": "user",
            "content": [
                {"audio": {"format": audio_format, "source": {"bytes": audio_bytes}}},
                {"text": prompt}
            ]
        }]
        
        inference_config = {
            "temperature": temperature,
            "maxTokens": max_tokens,
            "topP": top_p
        }
        
        try:
            response = self.bedrock.converse(
                modelId=get_current_model_id(),
                messages=messages,
                inferenceConfig=inference_config
            )
            
            if response and "output" in response:
                return response["output"]["message"]["content"][0]["text"]
            else:
                return "No response received"
                
        except Exception as e:
            return f"Error occurred: {str(e)}"

def load_audio_as_bytes(uploaded_file):
    """Load uploaded audio file as bytes"""
    try:
        audio_bytes = uploaded_file.read()
        return audio_bytes
    except Exception as e:
        return None
