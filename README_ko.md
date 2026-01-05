# Amazon Nova 2 Omni (Preview) 멀티모달 데모 & 벤치마크

Amazon Nova 2 Omni의 강력한 멀티모달 AI 기능을 체험하고 평가할 수 있는 Streamlit 기반 데모 애플리케이션 및 벤치마킹 도구입니다.

## 🚀 시작하기

### 사전 요구사항
- Python 3.12 이상
- Bedrock 서비스 액세스가 가능한 AWS 계정
- AWS 자격 증명 구성: `aws configure`

### 빠른 시작

1. **uv 설치** (아직 설치하지 않은 경우)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **프로젝트 클론**
   ```bash
   git clone https://github.com/daekeun-ml/nova-2-omni-examples
   cd nova-2-omni-examples
   ```

3. **의존성 설치**
   ```bash
   uv sync
   ```

4. **데모 실행**
   ```bash
   ./run_demo.sh
   ```
   
   또는 직접 실행:
   ```bash
   uv run streamlit run main.py
   ```

5. **브라우저에서 접속**
   - 로컬: http://localhost:8501
   - 사이드바에서 원하는 기능을 선택하여 탐색

## 🧪 오디오 벤치마킹

### STT (Speech-to-Text) 벤치마크

[Zeroth-Korean 데이터셋](https://huggingface.co/datasets/kresnik/zeroth_korean)을 사용하여 Amazon Nova 2 Omni의 한국어 음성 인식 성능을 벤치마크합니다.

**데이터셋 개요:**
- **총 데이터량**: 51.6시간의 훈련 데이터와 1.2시간의 테스트 데이터
- **발화 수**: 22,263개의 훈련 발화와 457개의 테스트 발화
- **화자 수**: 105명의 훈련 화자와 10명의 테스트 화자
- **샘플링 레이트**: 16kHz

**기능:**
- CER (Character Error Rate) 및 WER (Word Error Rate) 지표
- TTFT, End-to-End 지연시간의 P50/P95/P99 백분위수 측정
- 15개 동시 워커로 병렬 처리
- 실시간 진행률 표시
- JSON 형태로 상세 결과 저장

**벤치마크 실행:**
```bash
# 벤치마크 의존성 설치
uv sync --group benchmark

# 전체 457개 테스트 샘플에 대한 STT 벤치마크 실행
uv run benchmark_stt.py

# 기존 결과 분석
uv run benchmark_stt.py --analyze benchmark/benchmark_stt_results.json
```

**출력:**
- 콘솔: 요약 통계 (CER/WER 평균, 지연시간 백분위수)
- `benchmark/benchmark_stt_results.json`: 참조/예측 텍스트가 포함된 샘플별 상세 결과

---

## 🤖 Amazon Nova 2 Omni 소개

Amazon Nova 2 Omni는 Amazon의 차세대 멀티모달 추론 및 이미지 생성 모델입니다. 텍스트, 이미지, 비디오, 음성 입력을 지원하면서 텍스트와 이미지 출력을 생성하는 멀티모달 모델입니다.

### 🌟 주요 기능

#### 멀티모달 이해 및 생성
- **텍스트, 이미지, 비디오, 음성** 입력의 통합 처리
- **텍스트 및 이미지** 출력의 네이티브 생성
- 여러 AI 모델을 관리할 필요 없는 다양한 작업을 위한 단일 모델

#### 고급 추론 기능
- 대용량 문서 처리를 위한 **1M 토큰 컨텍스트 윈도우**
- 복잡한 다단계 추론 및 장기 계획
- 성능, 정확도, 비용 최적화를 위한 유연한 추론 제어

#### 언어 및 음성 지원
- 텍스트 처리에서 **200개 이상 언어** 지원
- 음성 입력에서 **10개 언어** 지원 (2026.01 Preview 기준)
- 다중 화자 대화 전사, 번역 및 요약

#### 이미지 생성 및 편집
- 자연어를 사용한 고품질 이미지 생성 및 편집
- 캐릭터 일관성 유지
- 이미지 내 텍스트 렌더링
- 객체 및 배경 수정 기능

#### 음성 이해
- 네이티브 추론을 통한 뛰어난 음성 이해
- 다중 화자 대화 전사, 번역 및 요약
- 실시간 고객 상호작용 지원

### 🏢 사용 사례

- **고객 서비스**: 멀티모달 챗봇 및 지원 시스템
- **콘텐츠 제작**: 마케팅 자료 및 광고 크리에이티브 제작
- **문서 분석**: 대용량 문서 및 비디오 콘텐츠 분석
- **음성 처리**: 회의 전사, 번역 및 요약
- **시각적 검색**: 이미지 및 비디오 기반 검색 시스템

---

## 참고 자료

Amazon Nova 2 Omni에 대한 자세한 정보는 [AWS 공식 문서](https://aws.amazon.com/nova/)를 참조하세요.

**참고**: 이 데모는 Amazon Nova 2 Omni의 기능을 체험하기 위해 제작되었습니다. 프로덕션 환경에서 사용하기 전에 적절한 보안 및 성능 최적화를 수행하세요.
