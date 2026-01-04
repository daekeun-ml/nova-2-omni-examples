# Amazon Nova 2 Omni (Preview) 멀티모달 데모

Amazon Nova 2 Omni의 강력한 멀티모달 AI 기능을 체험할 수 있는 Streamlit 기반 데모 애플리케이션입니다.

## 🚀 Getting Started

### 사전 요구사항
- Python 3.12 이상
- AWS 계정 및 Bedrock 서비스 액세스 권한
- AWS 자격 증명 설정: `aws configure`

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

6. **브라우저에서 접속**
   - 로컬: http://localhost:8501
   - 사이드바에서 원하는 기능 선택 후 체험

### 기존 pip 사용자
pip를 사용하는 경우:
```bash
pip install -r requirements.txt
streamlit run main.py
```

---

## 🤖 Amazon Nova 2 Omni 소개

Amazon Nova 2 Omni는 Amazon이 개발한 차세대 멀티모달 추론 및 이미지 생성 모델입니다. 텍스트, 이미지, 비디오, 음성 입력을 지원하며 텍스트와 이미지 출력을 모두 생성할 수 있는 멀티모달 모델입니다.

### 🌟 주요 특징

#### 멀티모달 이해 및 생성
- **텍스트, 이미지, 비디오, 음성** 입력을 통합적으로 처리
- **텍스트와 이미지** 출력을 네이티브하게 생성
- 여러 AI 모델을 관리할 필요 없이 하나의 모델로 다양한 작업 수행

#### 고급 추론 능력
- **1M 토큰 컨텍스트 윈도우**로 대용량 문서 처리
- 복잡한 다단계 추론 및 장기 계획 수립
- 유연한 추론 제어로 성능, 정확도, 비용 최적화

#### 언어 및 음성 지원
- **200개 이상 언어** 텍스트 처리 지원
- **10개 언어** 음성 입력 지원 (2026.01 Preview 기준)
- 다화자 대화 전사, 번역, 요약 기능

#### 이미지 생성 및 편집
- 자연어를 사용한 고품질 이미지 생성 및 편집
- 캐릭터 일관성 유지
- 이미지 내 텍스트 렌더링
- 객체 및 배경 수정 기능

#### 음성 이해
- 네이티브 추론을 통한 우수한 음성 이해
- 다화자 대화 전사, 번역, 요약
- 실시간 고객 상호작용 지원

### 🏢 활용 사례

- **고객 서비스**: 멀티모달 챗봇 및 지원 시스템
- **콘텐츠 생성**: 마케팅 자료 및 광고 크리에이티브 제작
- **문서 분석**: 대용량 문서 및 비디오 콘텐츠 분석
- **음성 처리**: 회의 전사, 번역, 요약
- **비주얼 검색**: 이미지 및 비디오 기반 검색 시스템

---

## References

Amazon Nova 2 Omni에 대한 자세한 정보는 [AWS 공식 문서](https://aws.amazon.com/nova/)를 참조하세요.

**주의사항**: 이 데모는 Amazon Nova 2 Omni의 기능을 체험하기 위한 목적으로 제작되었습니다. 프로덕션 환경에서 사용하기 전에 적절한 보안 및 성능 최적화를 수행하시기 바랍니다.
