import streamlit as st
import requests
import time
import io

# 페이지 설정
st.set_page_config(
    page_title="기타 TAB 생성기",
    page_icon="🎸",
    layout="wide"
)

# 메인 화면
st.title("🎸 기타 TAB 생성기")
st.markdown("---")

# API 서버 URL 설정
API_BASE_URL = "http://localhost:5000"

# 서버 상태 확인 함수
def check_server():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return response.status_code == 200
    except:
        return False

# 서버 상태 표시
if check_server():
    st.success("✅ API 서버 연결됨")
else:
    st.warning("⚠️ API 서버가 연결되지 않았습니다. Flask 서버를 먼저 실행해주세요.")

# 사이드바 설정--------------------------------------------------------------------------
# 입력 방법 선택
st.sidebar.title("🪧 입력 방법")
method = st.sidebar.selectbox("선택하세요:", ["YouTube 링크", "파일 업로드"])

# 매개변수 설정
st.sidebar.title("⚙️ 변환 설정")
threshold = st.sidebar.slider("음표 검출 임계값", 0.1, 0.9, 0.5, 0.1)
min_duration = st.sidebar.slider("최소 음표 길이 (초)", 0.01, 0.2, 0.05, 0.01)

st.sidebar.markdown("&nbsp;", unsafe_allow_html=True)

# 사용법
st.sidebar.title("📜 사용법")
st.sidebar.markdown("""
1. YouTube 링크: YouTube URL로부터 음원을 불러와 분석합니다
2. 파일 업로드: 로컬 WAV 파일을 업로드하여 분석합니다

**매개변수 설명:**
- 음표 검출 임계값: 낮을수록 더 많은 음표를 검출합니다 (0.1-0.9)
- 최소 음표 길이: 이보다 짧은 음표는 걸러냅니다 (0.01-0.2초)

**주의사항:**
- 정확도는 보장하지 않습니다
- WAV 형식만 지원하고 있습니다
- 변환에는 시간이 걸릴 수 있고, 긴 음원일수록 시간이 더 걸립니다
- 변환 후에는 초기화 버튼을 눌러야만 다음 프로젝트를 시작할 수 있습니다
""")
# ----------------------------------------------------------------------------------------

# 결과 출력을 위한 상태 초기화
if 'tab_result' not in st.session_state:
    st.session_state.tab_result = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

# YouTube 섹션
if method == "YouTube 링크":
    st.header("🔗 YouTube 음원 가져오기")
    
    url = st.text_input(
        "YouTube URL:",
        placeholder="https://www.youtube.com/watch?v=example"
    )
    
    if st.button("🎵 분석 시작", disabled=st.session_state.processing):
        if url:
            if not check_server():
                st.error("API 서버에 연결할 수 없습니다.")
            else:
                st.session_state.processing = True
                st.session_state.tab_result = None
                
                # 진행 상태 표시
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # YouTube URL 처리 요청
                    status_text.text("YouTube 음원 다운로드 중...")
                    progress_bar.progress(25)
                    
                    response = requests.post(
                        f"{API_BASE_URL}/convert_youtube",
                        json={
                            "url": url,
                            "threshold": threshold,
                            "min_duration": min_duration
                        },
                        timeout=300
                    )
                    
                    progress_bar.progress(100)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.tab_result = result.get('tab_content', '')
                        status_text.text("✅ 변환 완료!")
                    else:
                        error_msg = response.json().get('error', '알 수 없는 오류')
                        st.error(f"변환 실패: {error_msg}")
                        status_text.text("")
                        progress_bar.empty()
                        
                except requests.exceptions.Timeout:
                    st.error("변환 시간이 초과되었습니다. 짧은 음원으로 시도해 주세요.")
                    status_text.text("")
                    progress_bar.empty()
                except requests.exceptions.RequestException as e:
                    st.error(f"API 요청 중 오류 발생: {str(e)}")
                    status_text.text("")
                    progress_bar.empty()
                finally:
                    st.session_state.processing = False
        else:
            st.error("YouTube URL을 입력해주세요.")

# 파일 업로드
else:
    st.header("📁 파일 업로드")
    
    file = st.file_uploader(
        "WAV 오디오 파일 선택:",
        type=['wav'],
        help="WAV 형식의 오디오 파일만 지원됩니다."
    )
    
    if file:
        st.success(f"파일 선택됨: {file.name} ({file.size / 1024 / 1024:.1f} MB)")
        
        # 파일 정보 표시
        st.info(f"파일 크기: {file.size:,} bytes")
        
        if st.button("분석 시작", disabled=st.session_state.processing):
            if not check_server():
                st.error("API 서버에 연결할 수 없습니다.")
            else:
                st.session_state.processing = True
                st.session_state.tab_result = None
                
                # 진행 상태 표시
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # 파일 업로드 및 처리 요청
                    status_text.text("파일 업로드 중...")
                    progress_bar.progress(25)
                    
                    files = {'file': (file.name, file.getvalue(), 'audio/wav')}
                    data = {
                        'threshold': threshold,
                        'min_duration': min_duration
                    }
                    
                    status_text.text("WAV → MIDI 변환 중...")
                    progress_bar.progress(50)
                    
                    response = requests.post(
                        f"{API_BASE_URL}/convert_wav",
                        files=files,
                        data=data,
                        timeout=300
                    )
                    
                    progress_bar.progress(100)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.tab_result = result.get('tab_content', '')
                        status_text.text("✅ 변환 완료!")
                    else:
                        error_msg = response.json().get('error', '알 수 없는 오류')
                        st.error(f"변환 실패: {error_msg}")
                        status_text.text("")
                        progress_bar.empty()
                        
                except requests.exceptions.Timeout:
                    st.error("변환 시간이 초과되었습니다. 짧은 음원으로 시도해주세요.")
                    status_text.text("")
                    progress_bar.empty()
                except requests.exceptions.RequestException as e:
                    st.error(f"API 요청 중 오류 발생: {str(e)}")
                    status_text.text("")
                    progress_bar.empty()
                finally:
                    st.session_state.processing = False

# TAB 악보 출력
if st.session_state.tab_result:
    st.markdown("---")
    st.header("🎼 생성된 기타 TAB 악보")
    
    # TAB 악보를 코드 블록으로 표시
    st.code(st.session_state.tab_result, language=None)
    
    # 다운로드 버튼
    st.download_button(
        label="📥 생성된 악보 다운로드",
        data=st.session_state.tab_result,
        file_name="guitar_tab.txt",
        mime="text/plain"
    )
    
    # 초기화 버튼
    if st.button("🔄 다음 프로젝트를 위해 초기화하기"):
        st.session_state.tab_result = None
        st.rerun()

st.markdown("---")
st.markdown("""
            <div style='text-align: right;'>
                <p>인공지능응용/게임프로그래밍패턴 기말 프로젝트 | C077021 이동훈</p>
            </div>
            """,
            unsafe_allow_html=True)