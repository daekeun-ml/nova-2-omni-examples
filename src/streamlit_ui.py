import streamlit as st
from io import BytesIO
from PIL import Image

# 지연 임포트를 위한 함수들
def _import_common():
    from .common import (
        get_bedrock_runtime, get_current_model_id, translate_to_english, 
        translate_to_user_language, detect_non_english, convert_image_to_bytes, 
        load_image_as_bytes, call_nova_model, format_stt_result, translate_text, 
        extract_video_frames, parse_json_from_text
    )
    return locals()

def _import_pil():
    
    from PIL import Image
    return BytesIO, Image

def _import_analyzers():
    from .document_analysis import DocumentAnalyzer
    from .object_detection import ObjectDetector
    from .video_understanding import VideoAnalyzer, VIDEO_ANALYSIS_PROMPTS, get_video_format, parse_timestamps
    from .image_generation import ImageGenerator
    from .image_editing import ImageEditor
    return locals()

def _import_optional():
    try:
        import fitz
        from .multi_agent import create_safety_agent, create_coordinator_agent, MultiAgentOrchestrator
        from .speech_understanding import SpeechAnalyzer, load_audio_as_bytes
        return locals()
    except ImportError:
        return {}

def main():
    st.title("Amazon Nova 2 Omni 멀티모달 데모")
    st.markdown("이미지 생성, 편집, 비디오 이해, 문서 분석/OCR 기능, 객체 탐지, Mult-Agent 추론을 체험해 보세요!")
    
    with st.sidebar:
        # 기능 선택
        st.header("🎯 기능 선택")
        feature = st.selectbox(
            "사용할 기능을 선택하세요:",
            ["이미지 생성", "이미지 편집", "비디오 이해", "오디오 이해", "문서 분석 & OCR", "객체 탐지", "Multi-Agent 추론"]
        )
        
        st.divider()
        
        st.header("⚙️ 공통 설정")
        
        # Temperature 설정
        temperature = st.slider(
            "Temperature:",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="높을수록 더 창의적이고 다양한 결과"
        )
        
        # Top P 설정
        top_p = st.slider(
            "Top P:",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="낮을수록 더 집중된 결과, 높을수록 더 넓은 범위의 토큰 고려"
        )
        
        # Max tokens 설정
        max_tokens = st.slider(
            "최대 토큰 수:",
            min_value=100,
            max_value=4000,
            value=2000,
            step=100,
            help="생성할 텍스트의 최대 길이"
        )
        
        st.divider()
        
        # 모델 및 리전 설정
        st.header("🌐 모델 & 리전 설정")
        
        model_id = st.text_input(
            "모델 ID:",
            value="us.amazon.nova-2-omni-v1:0",
            help="사용할 Nova 모델 ID 입력"
        )
        
        region_id = st.selectbox(
            "AWS 리전:",
            ["us-west-2", "us-east-1"],
            index=0,
            help="AWS Bedrock 서비스 리전"
        )
        
        # 세션 상태에 저장
        st.session_state.model_id = model_id
        st.session_state.region_id = region_id
        
        st.divider()
        
        st.header("📋 사용 가이드")
        st.markdown("""
        **Note:**
        - 한국어/중국어/일본어 프롬프트는 자동으로 영어로 번역됩니다.
        - 이미지는 PNG 형식으로 다운로드 가능합니다.
        - 비디오는 1분 이하 분량을 권장합니다.
        - 오디오는 25MB 이하 용량을 권장합니다.
        """)
    
    # 사이드바에서 기능 선택
    if feature == "이미지 생성":
        image_generation_demo(temperature, max_tokens, top_p)
    elif feature == "이미지 편집":
        image_editing_demo(temperature, max_tokens, top_p)
    elif feature == "비디오 이해":
        video_understanding_demo(temperature, max_tokens, top_p)
    elif feature == "오디오 이해":
        speech_understanding_demo(temperature, max_tokens, top_p)
    elif feature == "문서 분석 & OCR":
        document_analysis_demo(temperature, max_tokens, top_p)
    elif feature == "객체 탐지":
        object_detection_demo(temperature, max_tokens, top_p)
    elif feature == "Multi-Agent 추론":
        multi_agent_demo(temperature, max_tokens, top_p)

def speech_understanding_demo(temperature, max_tokens, top_p):
    """음성 이해 데모"""
    st.header("🎙️ 음성 이해")
    
    st.markdown("""
    Amazon Nova 2 Omni를 사용하여 오디오 콘텐츠를 분석합니다.
    
    **지원 형식**: mp3, wav, aac, flac, ogg
    **권장 설정**: Temperature 0, TopP 1 (정확한 분석을 위해)
    """)
    
    uploaded_file = st.file_uploader(
        "오디오 파일을 업로드하세요:", 
        type=['mp3', 'wav', 'aac', 'flac', 'ogg'],
        help="최대 25MB 권장"
    )
    
    if uploaded_file:
        st.audio(uploaded_file)
        
        analysis_type_korean = st.selectbox(
            "분석 유형을 선택하세요:",
            ["STT (음성-텍스트 변환)", "STT + 번역", "통화 분석", "화자 분리", "요약", "감정 분석", "핵심 포인트", "질의응답"]
        )

        type_mapping = {
            "STT (음성-텍스트 변환)": "transcription",
            "STT + 번역": "translation",
            "통화 분석": "call_analytics",
            "화자 분리": "diarization", 
            "요약": "summary",
            "감정 분석": "sentiment",
            "핵심 포인트": "key_points",
            "질의응답": "qa"
        }
        
        analysis_type = type_mapping[analysis_type_korean]
        
        # Q&A인 경우 질문 입력
        question = None
        if analysis_type == "qa":
            question = st.text_input("질문을 입력하세요:", value="이 오디오의 주요 내용은 무엇인가요?")
        
        if st.button("오디오 분석 시작"):
            try:
                from .speech_understanding import load_audio_as_bytes
                audio_bytes = load_audio_as_bytes(uploaded_file)
                audio_format = uploaded_file.name.split('.')[-1].lower()
                
                if audio_bytes:
                    with st.spinner("오디오를 분석하는 중..."):
                        from .speech_understanding import SpeechAnalyzer, load_audio_as_bytes
                        analyzer = SpeechAnalyzer()
                        
                        # 오디오 분석 실행
                        result = analyzer.analyze_audio(
                            audio_bytes=audio_bytes,
                            audio_format=audio_format,
                            analysis_type=analysis_type,
                            question=question,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p
                        )
                        
                        if result and result != "No response received":
                            st.success("✅ 오디오 분석 완료!")
                            
                            # STT와 화자 분리는 원래 언어 그대로, 나머지는 한국어로 번역
                            if analysis_type in ["transcription", "diarization"]:
                                st.subheader(f"📋 {analysis_type_korean} 결과")
                                # STT 결과 가독성 향상
                                from .common import format_stt_result
                                formatted_result = format_stt_result(result)
                                st.markdown(formatted_result)
                            else:
                                # 결과를 한국어로 번역
                                st.write("🔄 결과를 한국어로 번역 중...")
                                from .common import translate_to_user_language
                                korean_result = translate_to_user_language(result)
                                
                                st.subheader(f"📋 {analysis_type_korean} 결과")
                                st.markdown(korean_result)
                        else:
                            st.error("API 응답을 받지 못했습니다.")
                else:
                    st.error("오디오 파일을 읽을 수 없습니다.")
                    
            except NameError:
                st.error("Speech Understanding 모듈을 불러올 수 없습니다. 필요한 패키지를 설치해주세요.")
            except Exception as e:
                st.error(f"분석 중 오류 발생: {str(e)}")
    else:
        st.info("👆 오디오 파일을 업로드하여 분석을 시작하세요.")

def image_generation_demo(temperature, max_tokens, top_p):
    """이미지 생성 데모"""
    st.header("🎨 이미지 생성")
    st.markdown("""
    텍스트 프롬프트로 이미지를 생성합니다.
    
    **텍스트-이미지 생성:**
    - 최대 출력 크기: 4,194,304 픽셀 (4 메가픽셀)
    - 기본 비율: 16:9
    - 가로형: 2:1 (2880x1440), 16:9 (2704x1520), 3:2 (2496x1664), 4:3 (2352x1760)
    - 정사각형: 1:1 (2048x2048)
    - 세로형: 1:2 (1440x2880), 9:16 (1520x2704), 2:3 (1664x2496), 3:4 (1760x2352)
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        prompt = st.text_area(
            "이미지 생성 프롬프트를 입력하세요:",
            value="밤하늘 아래 아늑한 캠프파이어의 이미지를 만들어주세요",
            height=100
        )
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            aspect_ratio = st.selectbox(
                "화면 비율:",
                ["16:9 (기본)", "2:1 (와이드)", "3:2 (가로)", "4:3 (가로)", 
                 "1:1 (정사각형)", "1:2 (세로)", "9:16 (세로)", "2:3 (세로)", "3:4 (세로)"]
            )
            
            visual_style = st.selectbox(
                "비주얼 스타일:",
                ["사실적 이미지 (기본)", "스토리북 일러스트", "애니메이션 (2D - 일본풍)", "애니메이션 (2D - 서양풍)", "애니메이션 (3D)", "디지털 아트", 
                 "수채화", "유화", "만화/카툰", "미니멀", "빈티지"]
            )
        
        with col_b:
            temperature = st.slider(
                "Temperature:",
                min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                help="0에 가까울수록 일관된 결과, 1에 가까울수록 창의적인 결과"
            )
            
            max_tokens = st.slider(
                "최대 토큰 수:",
                min_value=1000, max_value=10000, value=4000, step=500,
                help="생성할 수 있는 최대 토큰 수"
            )
            
            reasoning = st.checkbox(
                "추론 모드 활성화",
                help="더 정교한 이미지 생성을 위한 추론 과정 활성화"
            )
        
        if st.button("이미지 생성", type="primary"):
            try:
                with st.spinner("이미지를 생성하는 중..."):
                    # 필요할 때만 임포트
                    from .image_generation import ImageGenerator
                    generator = ImageGenerator()
                    
                    result = generator.generate_image(
                        prompt=prompt,
                        visual_style=visual_style,
                        aspect_ratio=aspect_ratio,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        reasoning=reasoning
                    )
                    
                    if result["success"]:
                        st.session_state.generated_image = result["image"]
                        st.success("✅ 이미지 생성 완료!")
                        st.write(f"🎨 스타일: {result['style']}")
                        st.write(f"🎯 최종 프롬프트: {result['prompt']}")
                    else:
                        st.error(f"❌ {result['error']}")
                        
            except Exception as e:
                st.error(f"이미지 생성 중 오류 발생: {str(e)}")
    
    with col2:
        if "generated_image" in st.session_state:
            st.image(st.session_state.generated_image, caption="생성된 이미지")
            
            # 다운로드 버튼 추가
            
            img_buffer = BytesIO()
            st.session_state.generated_image.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()
            
            st.download_button(
                label="📥 이미지 다운로드",
                data=img_bytes,
                file_name="generated_image.png",
                mime="image/png",
                type="secondary"
            )

def image_editing_demo(temperature, max_tokens, top_p):
    """이미지 편집 데모"""
    st.header("✏️ 이미지 편집")
    st.markdown("""
    기존 이미지에 요소를 추가하거나 수정합니다.
    
    **이미지 편집:**
    - 최대 출력 크기: 1,048,576 픽셀 (1 메가픽셀)
    - 출력은 입력 이미지의 비율과 동일 (별도 요청 시 제외)
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 기본 이미지 사용 옵션
        use_default = st.checkbox("기본 이미지 사용 (img-editing.png)", value=True)
        
        if use_default:
            try:
                default_path = "samples/img-editing.png"
                
                with open(default_path, "rb") as f:
                    uploaded_file = BytesIO(f.read())
                    uploaded_file.name = "img-editing.png"
                image = Image.open(default_path)
                st.image(image, caption="기본 이미지 (img-editing.png)")
            except FileNotFoundError:
                st.error("기본 이미지를 찾을 수 없습니다. 이미지를 업로드해주세요.")
                uploaded_file = st.file_uploader("편집할 이미지를 업로드하세요", type=['png', 'jpg', 'jpeg'])
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="업로드된 이미지")
        else:
            uploaded_file = st.file_uploader("편집할 이미지를 업로드하세요", type=['png', 'jpg', 'jpeg'])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="업로드된 이미지")
        
        if uploaded_file:
            
            edit_type = st.selectbox(
                "편집 유형:",
                ["사용자 정의", "텍스트 추가", "사물/인물 추가", "사물/인물 제거", "배경 변경", "색상 변경", "스타일 변경"]
            )
            
            # 편집 유형별 파라미터 수집
            edit_params = {"edit_type": edit_type}
            
            if edit_type == "사용자 정의":
                edit_params["edit_prompt"] = st.text_area(
                    "편집 지시사항을 입력하세요:",
                    value="이미지 왼쪽 하단에 사자를 추가해주세요",
                    height=100
                )
            elif edit_type == "텍스트 추가":
                edit_params["text_content"] = st.text_input("추가할 텍스트:", value="Amazon")
                edit_params["text_position"] = st.text_area("텍스트 위치 및 추가 설명:", value="가운데 빌딩 유리에. 그냥 오버레이가 아니라 빌딩 유리창 장식이야. 글자가 너무 크면 안되겠지? ", height=60)
                edit_params["text_style"] = st.selectbox("텍스트 스타일:", ["간판", "유리창 글씨", "벽면 페인팅", "네온사인", "조각/새김"])
            elif edit_type == "사물/인물 추가":
                edit_params["object_to_add"] = st.text_input("추가할 사물/인물:", value="고양이")
                edit_params["add_position"] = st.selectbox("추가 위치:", ["왼쪽", "오른쪽", "중앙", "배경", "전경"])
                edit_params["integration_style"] = st.selectbox("통합 방식:", ["자연스럽게", "사실적으로", "조화롭게"])
            elif edit_type == "사물/인물 제거":
                edit_params["object_to_remove"] = st.text_input("제거할 사물/인물:", value="호랑이")
            elif edit_type == "배경 변경":
                edit_params["new_background"] = st.text_input("새로운 배경:", value="바다")
                edit_params["transition_style"] = st.selectbox("전환 방식:", ["자연스럽게", "완전히 교체", "부분적으로"])
            elif edit_type == "색상 변경":
                edit_params["target_object"] = st.text_input("색상을 바꿀 대상:", value="호랑이")
                edit_params["new_color"] = st.text_input("새로운 색상:", value="흰색")
            elif edit_type == "스타일 변경":
                edit_params["new_style"] = st.selectbox("새로운 스타일:", ["애니메이션 (2D - 일본풍)", "애니메이션 (2D - 서양풍)", "애니메이션 (3D)", "수채화", "유화", "만화", "빈티지"])
            
            if st.button("이미지 편집", type="primary"):
                try:
                    with st.spinner("이미지를 편집하는 중..."):
                        from .image_editing import ImageEditor
                        editor = ImageEditor()
                        
                        result = editor.edit_image(
                            uploaded_file=uploaded_file,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            **edit_params
                        )
                        
                        if result["success"]:
                            st.session_state.edited_image = result["image"]
                            st.success("✅ 이미지 편집 완료!")
                            st.write(f"📝 편집 프롬프트: {result['prompt']}")
                        else:
                            st.error(f"❌ {result['error']}")
                            
                except Exception as e:
                    st.error(f"이미지 편집 중 오류 발생: {str(e)}")
    
    with col2:
        if "edited_image" in st.session_state:
            st.image(st.session_state.edited_image, caption="편집된 이미지")
            
            # 다운로드 버튼 추가
            
            img_buffer = BytesIO()
            st.session_state.edited_image.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()
            
            st.download_button(
                label="📥 편집된 이미지 다운로드",
                data=img_bytes,
                file_name="edited_image.png",
                mime="image/png",
                type="secondary"
            )

def video_understanding_demo(temperature, max_tokens, top_p):
    """비디오 이해 데모"""
    st.header("🎬 비디오 이해")
    
    st.markdown("""
    Amazon Nova 2 Omni를 사용하여 비디오 콘텐츠를 분석합니다.
    
    **지원 형식**: mp4, mov, avi, mkv, webm
    **권장 설정**: Temperature 0, TopP 1 (정확한 분석을 위해)
    """)
    
    uploaded_file = st.file_uploader(
        "비디오 파일을 업로드하세요:", 
        type=["mp4", "mov", "avi", "mkv", "webm"],
        help="30초 이하의 비디오를 권장합니다"
    )
    
    if uploaded_file:
        st.video(uploaded_file)
        
        analysis_type = st.selectbox(
            "분석 유형:",
            ["요약", "하이라이트 추출", "시각적 설명", "이벤트 타임스탬프", "비디오 세그멘테이션", "비디오 분류", "사용자 정의"]
        )
        
        custom_prompt = None
        event_query = None
        
        if analysis_type == "사용자 정의":
            custom_prompt = st.text_area(
                "분석 프롬프트를 입력하세요:",
                value="What can you see in this video?",
                height=100
            )
        elif analysis_type == "이벤트 타임스탬프":
            event_query = st.text_input(
                "감지할 이벤트를 입력하세요:",
                value="mixing ingredients",
                help="예: mixing ingredients, adding sugar, cutting vegetables"
            )
        
        if st.button("비디오 분석 시작"):
            try:
                from .video_understanding import get_video_format
                video_bytes = uploaded_file.read()
                video_format = get_video_format(uploaded_file.name)
                
                with st.spinner("비디오를 분석하는 중..."):
                    from .video_understanding import VideoAnalyzer, VIDEO_ANALYSIS_PROMPTS, get_video_format, parse_timestamps
                    analyzer = VideoAnalyzer()
                    
                    # 프롬프트 준비
                    if analysis_type == "사용자 정의":
                        prompt = translate_to_english(custom_prompt) if detect_non_english(custom_prompt) else custom_prompt
                    elif analysis_type == "이벤트 타임스탬프":
                        translated_event = translate_to_english(event_query) if detect_non_english(event_query) else event_query
                        prompt = VIDEO_ANALYSIS_PROMPTS[analysis_type](translated_event)
                    else:
                        prompt = VIDEO_ANALYSIS_PROMPTS[analysis_type]
                    
                    result = analyzer.analyze_video(video_bytes, video_format, prompt, temperature, top_p, max_tokens)
                    
                    if isinstance(result, dict) and 'output' in result:
                        result_text = result['output']['message']['content'][0]['text']
                    else:
                        result_text = str(result)
                    
                    st.success("✅ 비디오 분석 완료!")
                    st.subheader(f"📋 {analysis_type} 결과")
                    
                    # 하이라이트 추출 특별 처리
                    if analysis_type == "하이라이트 추출":
                        uploaded_file.seek(0)
                        video_bytes = uploaded_file.read()
                        
                        highlight_result = analyzer.process_highlights(result_text, video_bytes)
                        
                        if highlight_result["success"]:
                            st.subheader("🎯 하이라이트 프레임")
                            cols = st.columns(min(3, len(highlight_result["highlight_frames"])))
                            
                            for idx, highlight in enumerate(highlight_result["highlight_frames"]):
                                with cols[idx % 3]:
                                    st.image(highlight["frame"], caption=f"⏰ {highlight['timestamp']:.1f}초", width=200)
                                    st.write(f"🔥 임팩트: {highlight['impact']}")
                                    if highlight['keywords']:
                                        st.write(f"🏷️ 키워드: {', '.join(highlight['keywords'])}")
                                    st.write(f"📝 {highlight['description']}")
                                    st.write("---")
                            
                            st.download_button(
                                label="📥 하이라이트 프레임 ZIP 다운로드",
                                data=highlight_result["zip_data"],
                                file_name="video_highlights.zip",
                                mime="application/zip",
                                type="secondary"
                            )
                        else:
                            st.error(highlight_result["error"])
                            st.markdown(result_text)
                    
                    # 이벤트 타임스탬프 특별 처리
                    elif analysis_type == "이벤트 타임스탬프":
                        st.markdown(result_text)
                        from .video_understanding import parse_timestamps
                        timestamps = parse_timestamps(result_text)
                        if timestamps:
                            uploaded_file.seek(0)
                            video_bytes = uploaded_file.read()
                            for i, (start, end) in enumerate(timestamps, 1):
                                st.write(f"**이벤트 {i}**: {start:.1f}초 - {end:.1f}초")
                                frames = extract_video_frames(video_bytes, [start, end])
                                if len(frames) >= 2:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.image(frames[0], caption=f"시작: {start:.1f}초", width=250)
                                    with col2:
                                        st.image(frames[1], caption=f"끝: {end:.1f}초", width=250)
                    else:
                        # 다른 분석 유형은 한국어로 번역
                        korean_result = translate_to_user_language(result_text)
                        st.markdown(korean_result)
                        st.session_state.video_result = korean_result
                    
            except Exception as e:
                st.error(f"비디오 분석 중 오류 발생: {str(e)}")
    
    # 이전 결과 표시
    if "video_result" in st.session_state:
        st.markdown("---")
        st.subheader("📄 이전 분석 결과")
        st.markdown(st.session_state.video_result)

def document_analysis_demo(temperature, max_tokens, top_p):
    """문서 분석 및 OCR 데모"""
    st.header("📄 문서 분석 & OCR")
    st.markdown("PDF 문서나 이미지에서 텍스트를 추출하고 분석합니다.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "문서를 업로드하세요", 
            type=['pdf', 'png', 'jpg', 'jpeg']
        )
        
        if uploaded_file:
            file_type = uploaded_file.type
            
            if "pdf" in file_type:
                st.success("PDF 파일이 업로드되었습니다.")
            else:
                image = Image.open(uploaded_file)
                st.image(image, caption="업로드된 이미지")
            
            analysis_option = st.selectbox(
                "분석 옵션:",
                ["OCR (텍스트 추출)", "핵심 정보 추출", "표 데이터 추출", "문서 요약"]
            )
            
            if st.button("문서 분석", type="primary"):
                with st.spinner("문서를 분석하는 중..."):
                    try:
                        from .document_analysis import DocumentAnalyzer
                        analyzer = DocumentAnalyzer()
                        file_bytes = uploaded_file.read()
                        
                        if "pdf" in file_type:
                            # PDF 병렬 처리
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            def progress_callback(progress):
                                if isinstance(progress, str):
                                    status_text.text(progress)
                                else:
                                    progress_bar.progress(progress)
                            
                            result = analyzer.analyze_pdf_parallel(
                                file_bytes=file_bytes,
                                analysis_option=analysis_option,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                top_p=top_p,
                                progress_callback=progress_callback
                            )
                        else:
                            # 이미지 처리
                            result = analyzer.analyze_image(
                                file_bytes=file_bytes,
                                analysis_option=analysis_option,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                top_p=top_p
                            )
                        
                        if result:
                            st.session_state.document_result = result
                        else:
                            st.error("분석 결과를 받지 못했습니다.")
                            
                    except ImportError as e:
                        st.error(f"모듈 로드 오류: {str(e)}")
                    except Exception as e:
                        st.error(f"분석 중 오류 발생: {str(e)}")
    
    with col2:
        if "document_result" in st.session_state:
            st.markdown("### 분석 결과:")
            st.markdown(st.session_state.document_result)

def object_detection_demo(temperature, max_tokens, top_p):
    """객체 탐지 데모"""
    st.header("🔍 객체 탐지")
    st.markdown("이미지에서 객체를 탐지하고 bounding box로 표시합니다.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 기본 이미지 사용 옵션
        use_default = st.checkbox("기본 이미지 사용 (img-car.png)", value=True)
        
        if use_default:
            try:
                default_path = "samples/img-car.png"
                with open(default_path, "rb") as f:
                    uploaded_file = BytesIO(f.read())
                    uploaded_file.name = "img-car.png"
                image = Image.open(default_path)
                st.image(image, caption="기본 이미지 (img-car.png)")
            except FileNotFoundError:
                st.error("기본 이미지를 찾을 수 없습니다. 이미지를 업로드해주세요.")
                uploaded_file = st.file_uploader("분석할 이미지를 업로드하세요", type=['png', 'jpg', 'jpeg'])
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="업로드된 이미지")
        else:
            uploaded_file = st.file_uploader("분석할 이미지를 업로드하세요", type=['png', 'jpg', 'jpeg'])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="업로드된 이미지")
        
        if uploaded_file:
            
            detection_type = st.selectbox(
                "탐지 유형:",
                ["모든 객체", "사람", "차량", "동물", "음식 & 음료", "전자제품", "가구", "스포츠", "도구", "식물", "사용자 정의"]
            )
            
            custom_object = None
            if detection_type == "사용자 정의":
                custom_object = st.text_input("탐지할 객체를 입력하세요 (영어):", value="cat")
            
            if st.button("객체 탐지", type="primary"):
                with st.spinner("객체를 탐지하는 중..."):
                    try:
                        from .object_detection import ObjectDetector
                        detector = ObjectDetector()
                        
                        result = detector.detect_objects(
                            image=image,
                            detection_type=detection_type,
                            custom_object=custom_object,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p
                        )
                        
                        if result:
                            # 전처리 메시지 표시
                            if result.get("processing_message"):
                                st.info(result["processing_message"])
                            
                            st.session_state.detection_image = result["annotated_image"]
                            st.session_state.detection_json = result["detection_json"]
                            st.session_state.detection_summary = f"🎯 탐지된 객체 수: {result['bbox_count']}\n\n📏 원본 이미지 크기: {result['original_size'][0]} x {result['original_size'][1]}"
                            st.rerun()
                        else:
                            st.error("객체 탐지 결과를 받지 못했습니다.")
                            
                    except ImportError as e:
                        st.error(f"모듈 로드 오류: {str(e)}")
                    except Exception as e:
                        st.error(f"객체 탐지 중 오류 발생: {str(e)}")
    
    with col2:
        if "detection_summary" in st.session_state:
            st.markdown("### 탐지 결과:")
            st.write(st.session_state.detection_summary)
            
            # JSON 데이터를 클릭해서 보기
            if "detection_json" in st.session_state and st.session_state.detection_json:
                with st.expander("📋 상세 정보 (클릭해서 보기)"):
                    st.json(st.session_state.detection_json)
        
        # 탐지된 이미지 표시 (bounding box 포함)
        if "detection_image" in st.session_state:
            st.markdown("### 탐지된 객체 (Bounding Box):")
            st.image(st.session_state.detection_image, caption="객체 탐지 결과")
            
            # 다운로드 버튼 추가
            img_buffer = BytesIO()
            st.session_state.detection_image.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()
            
            st.download_button(
                label="📥 탐지 결과 다운로드",
                data=img_bytes,
                file_name="object_detection_result.png",
                mime="image/png",
                type="secondary"
            )

def multi_agent_demo(temperature, max_tokens, top_p):
    """Multi-Agent 멀티모달 추론 데모"""
    st.header("🤖 Multi-Agent 멀티모달 추론")
    
    st.markdown("""
    여러 전문 에이전트가 협력하여 이미지를 종합적으로 분석합니다:
    - **안전성 분석 에이전트**: 위험 요소 식별 및 안전 조치 권장
    - **코디네이터 에이전트**: 모든 분석 결과를 종합하여 최종 보고서 작성
    """)
    
    tab1, tab2 = st.tabs(["🔧 Multi-Agent 설정", "📊 이미지 분석"])
    
    with tab1:
        st.subheader("Multi-Agent 전용 설정")
        st.info("💡 Multi-Agent 추론은 더 많은 토큰을 사용합니다. 필요에 따라 설정을 조정하세요.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Reasoning 모드 설정
            reasoning_mode = st.selectbox(
                "🧠 Reasoning 모드:",
                ["off", "low", "medium", "high"],
                index=2,  # 기본값: medium
                help="높을수록 더 깊이 있는 추론. high일 때는 Temperature/TopP 설정이 무시될 수 있습니다."
            )
            
            # Temperature 오버라이드
            use_custom_temp = st.checkbox("Temperature 오버라이드", value=False)
            if use_custom_temp:
                ma_temperature = st.slider(
                    "Multi-Agent Temperature:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.1,
                    help="Multi-Agent 전용 Temperature 설정"
                )
            else:
                ma_temperature = temperature
        
        with col2:
            # Max Tokens 오버라이드
            use_custom_tokens = st.checkbox("Max Tokens 오버라이드", value=True)
            if use_custom_tokens:
                ma_max_tokens = st.slider(
                    "Multi-Agent Max Tokens:",
                    min_value=1000,
                    max_value=8000,
                    value=4000,
                    step=500,
                    help="Multi-Agent 전용 최대 토큰 수"
                )
            else:
                ma_max_tokens = max_tokens
            
            # Top P 오버라이드
            use_custom_top_p = st.checkbox("Top P 오버라이드", value=False)
            if use_custom_top_p:
                ma_top_p = st.slider(
                    "Multi-Agent Top P:",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.9,
                    step=0.1,
                    help="Multi-Agent 전용 Top P 설정"
                )
            else:
                ma_top_p = top_p
        
        # 현재 설정 요약
        st.subheader("📋 현재 Multi-Agent 설정")
        settings_col1, settings_col2 = st.columns(2)
        with settings_col1:
            st.metric("Reasoning 모드", reasoning_mode)
            st.metric("Temperature", f"{ma_temperature:.1f}")
        with settings_col2:
            st.metric("Max Tokens", f"{ma_max_tokens:,}")
            st.metric("Top P", f"{ma_top_p:.1f}")
    
    with tab2:
        st.subheader("이미지 업로드 및 분석")
        
        # 기본 이미지 사용 옵션
        use_default = st.checkbox("기본 이미지 사용 (img-car.png)", value=True)
        
        if use_default:
            try:
                default_path = "samples/img-car.png"
                image = Image.open(default_path)
                st.image(image, caption="기본 이미지 (img-car.png)", width="stretch")
                # BytesIO 객체로 변환하여 uploaded_file처럼 사용
                with open(default_path, "rb") as f:
                    uploaded_file = BytesIO(f.read())
                    uploaded_file.name = "img-car.png"
            except FileNotFoundError:
                st.error("기본 이미지를 찾을 수 없습니다. 이미지를 업로드해주세요.")
                uploaded_file = st.file_uploader("분석할 이미지를 업로드하세요:", type=["png", "jpg", "jpeg"])
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="업로드된 이미지", width="stretch")
        else:
            uploaded_file = st.file_uploader("분석할 이미지를 업로드하세요:", type=["png", "jpg", "jpeg"])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="업로드된 이미지", width="stretch")

        if uploaded_file:
            if st.button("Multi-Agent 분석 시작"):
                try:
                    from .common import convert_image_to_bytes
                    image_bytes, image_format = convert_image_to_bytes(image)

                    with st.spinner("에이전트들이 분석 중입니다..."):
                        from .multi_agent import create_safety_agent, create_coordinator_agent, MultiAgentOrchestrator
                        # 에이전트 생성 (reasoning 모드 적용)
                        safety_agent = create_safety_agent(reasoning_mode)
                        coordinator_agent = create_coordinator_agent(reasoning_mode)

                        orchestrator = MultiAgentOrchestrator(
                            agents={"safety": safety_agent},
                            coordinator=coordinator_agent
                        )

                        tasks = {
                            "safety": [
                                {"image": {"format": image_format, "source": {"bytes": image_bytes}}},
                                {"text": "Analyze this image for safety risks. Identify all hazards, evaluate the overall risk level, and recommend appropriate safety actions."}
                            ]
                        }

                        st.write("=== Starting Multi-Agent Analysis ===")
                        st.write(f"⚙️ 설정: Temperature={ma_temperature}, MaxTokens={ma_max_tokens:,}, TopP={ma_top_p}, Reasoning={reasoning_mode.title()}")

                        # Multi-Agent 분석 실행 (오버라이드된 설정 사용)
                        result = orchestrator.run(tasks, ma_temperature, ma_max_tokens, ma_top_p)

                        st.write("=== Analysis Complete ===")
                        st.success("✅ Multi-Agent 분석 완료!")

                        if result and isinstance(result, dict):
                            if "summary" in result:
                                st.subheader("📋 종합 요약")
                                st.markdown(result["summary"])

                            if "key_insights" in result and result["key_insights"]:
                                st.subheader("💡 주요 통찰")
                                for i, insight in enumerate(result["key_insights"], 1):
                                    st.write(f"{i}. {insight}")

                            if "recommendations" in result and result["recommendations"]:
                                st.subheader("📌 권장사항")
                                for i, rec in enumerate(result["recommendations"], 1):
                                    st.write(f"{i}. {rec}")

                            with st.expander("🔍 상세 분석 결과 (JSON)"):
                                st.json(result)
                        else:
                            st.error("예상치 못한 결과 형식입니다.")
                            st.write("결과:", result)

                except NameError:
                    st.error("Multi-Agent 모듈을 불러올 수 없습니다. 필요한 패키지를 설치해주세요: pip install langchain-core pydantic")
                except Exception as e:
                    st.error(f"Multi-Agent 분석 중 오류가 발생했습니다: {str(e)}")
                    st.write("오류 상세:", str(e))
        else:
            st.info("👆 이미지를 업로드하여 Multi-Agent 분석을 시작하세요.")

if __name__ == "__main__":
    main()
