# Amazon Nova 2 Omni (Preview) 멀티모달 데모 & 벤치마크

Amazon Nova 2 Omni의 강력한 멀티모달 AI 기능을 체험하고 평가할 수 있는 Streamlit 기반 데모 애플리케이션 및 종합 벤치마킹 도구입니다.

[English README](README.md)

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

## 🧪 벤치마킹

### OCR 벤치마킹 (OCRBench v2)

[OCRBench v2 데이터셋](https://huggingface.co/datasets/ling99/OCRBench_v2)을 사용한 다양한 메트릭으로 OCR 성능을 종합 평가합니다.

**지원 메트릭:**
- **Text Accuracy**: 기본 텍스트 매칭 정확도
- **TEDS**: 테이블 편집 거리 기반 유사도 (테이블 파싱 태스크용)
- **IoU**: 교집합 대 합집합 비율 (위치 인식 태스크용)
- **VQA ANLS**: ANLS 점수를 사용한 VQA 태스크 평가
- **BLEU**: 기계번역 품질 메트릭 (OCR 태스크용)
- **F-measure**: 정밀도와 재현율의 조화평균 (OCR 태스크용)
- **ANLS**: 평균 정규화 레벤슈타인 유사도

**OCR 벤치마크 실행:**
```bash
# 벤치마크 의존성 설치
uv sync --group benchmark

# 100개 샘플로 실행 (기본값)
uv run python benchmark_ocr.py

# 특정 샘플 수로 실행
uv run python benchmark_ocr.py --num_samples 200

# 태스크 타입별 필터링으로 특정 메트릭 테스트
uv run python benchmark_ocr.py --num_samples 50 --task_filter "table"  # TEDS 메트릭
uv run python benchmark_ocr.py --num_samples 50 --task_filter "ocr"    # BLEU, F-measure
uv run python benchmark_ocr.py --num_samples 50 --task_filter "agent"  # IoU 메트릭
uv run python benchmark_ocr.py --num_samples 50 --task_filter "vqa"    # VQA ANLS
```

**출력:**
- 콘솔: 실시간 진행상황 및 종합 메트릭
- `benchmark/benchmark_ocr_results.json`: 샘플별 상세 결과

### STT 벤치마킹 (한국어)

[Zeroth-Korean 데이터셋](https://huggingface.co/datasets/kresnik/zeroth_korean)을 사용하여 Amazon Nova 2 Omni의 한국어 음성 인식 성능을 벤치마크합니다.

**데이터셋 개요:**
- **전체 데이터**: 훈련 데이터 51.6시간, 테스트 데이터 1.2시간
- **발화**: 훈련 발화 22,263개, 테스트 발화 457개
- **화자**: 훈련 화자 105명, 테스트 화자 10명
- **샘플링 레이트**: 16kHz

**기능:**
- CER (문자 오류율) 및 WER (단어 오류율) 메트릭
- P50/P95/P99 백분위수를 포함한 지연시간 측정 (TTFT, End-to-End)
- 15개 동시 워커를 사용한 병렬 처리
- 실시간 진행률 표시줄을 통한 진행상황 추적

**STT 벤치마크 실행:**
```bash
# 전체 457개 테스트 샘플로 STT 벤치마크 실행
uv run python benchmark_stt.py

# 특정 샘플 수로 실행
uv run python benchmark_stt.py --num_samples 100

# 기존 결과 분석
uv run python benchmark_stt.py --analyze benchmark/benchmark_stt_results.json
```

**출력:**
- 콘솔: 요약 통계 (CER/WER 평균, 지연시간 백분위수)
- `benchmark/benchmark_stt_results.json`: 샘플별 상세 결과

## 📊 벤치마크 구성

두 벤치마크 모두 다음 구성을 지원합니다:
- **동시 워커**: 15개 (병렬 처리용)
- **모델**: us.amazon.nova-2-omni-v1:0
- **리전**: us-west-2
- **메트릭**: 태스크별 평가 메트릭
- **진행상황 추적**: 실시간 tqdm 진행률 표시줄

## 🤖 Amazon Nova 2 Omni 소개

Amazon Nova 2 Omni는 Amazon의 차세대 멀티모달 추론 및 이미지 생성 모델입니다. 텍스트, 이미지, 비디오, 음성 입력을 지원하면서 텍스트와 이미지 출력을 생성하는 멀티모달 모델입니다.

### 🌟 주요 기능

#### 멀티모달 이해 및 생성
- **텍스트, 이미지, 비디오, 음성** 입력의 통합 처리
- **텍스트 및 이미지** 출력의 네이티브 생성
- 여러 AI 모델을 관리할 필요 없이 다양한 태스크를 위한 단일 모델

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

## 📁 프로젝트 구조

```
nova-2-omni-examples/
├── main.py                    # Streamlit 데모 애플리케이션
├── benchmark_ocr.py           # OCR 벤치마크 (OCRBench v2)
├── benchmark_stt.py           # STT 벤치마크 (Zeroth-Korean)
├── src/
│   ├── common.py             # 공통 유틸리티 및 구성
│   └── eval_metrics/
│       └── ocr_metrics.py    # OCR 평가 메트릭
├── benchmark/                # 벤치마크 결과 디렉토리
└── README.md                 # 이 파일
```

## 🔧 개발

### 새로운 벤치마크 추가

1. 기존 패턴을 따라 새로운 벤치마크 스크립트 생성
2. `src/eval_metrics/`에 평가 메트릭 추가
3. `pyproject.toml`에서 의존성 업데이트
4. README에 문서 추가

### 구성

- 모델 및 리전 설정: `src/common.py`
- 벤치마크 매개변수: 명령줄 인수
- 의존성: `pyproject.toml`

---

## 참고 자료

Amazon Nova 2 Omni에 대한 자세한 정보는 [AWS 공식 문서](https://aws.amazon.com/nova/)를 참조하세요.

**참고**: 이 데모는 Amazon Nova 2 Omni의 기능을 체험하기 위해 제작되었습니다. 프로덕션 환경에서 사용하기 전에 적절한 보안 및 성능 최적화를 수행하세요.
