import os
import sys
import tempfile
import traceback
import logging
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import yt_dlp

from wav_to_midi import WAVToMIDIConverter
from midi_to_tab import MidiToTabConverter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

app = Flask(__name__)
CORS(app)  # CORS 허용

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "best_model.pth"
TEMP_DIR = tempfile.gettempdir()

class TabGeneratorAPI:
    """TAB 생성 API 클래스"""
    
    def __init__(self, model_path):
        """
        Args:
            model_path (str): 학습된 모델 파일 경로
        """
        self.model_path = model_path
        self.wav_converter = None
        self.tab_converter = None
        
        # 모델 초기화
        self._initialize_converters()
    
    def _initialize_converters(self):
        try:
            if os.path.exists(self.model_path):
                logger.info(f"모델 로딩 중: {self.model_path}")
                self.wav_converter = WAVToMIDIConverter(self.model_path)
                logger.info("WAV to MIDI 변환기 초기화 완료")
            else:
                logger.warning(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
                
            self.tab_converter = MidiToTabConverter()
            logger.info("MIDI to TAB 변환기 초기화 완료")
            
        except Exception as e:
            logger.error(f"변환기 초기화 실패: {e}")
            raise
    
    def download_youtube_audio(self, url, output_path):
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': output_path,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'postprocessor_args': [
                    '-ar', '22050'
                ],
                'prefer_ffmpeg': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"YouTube 다운로드 시작: {url}")
                ydl.download([url])
                logger.info("YouTube 다운로드 완료")
                
        except Exception as e:
            logger.error(f"YouTube 다운로드 실패: {e}")
            raise
    
    def convert_wav_to_tab(self, wav_path, threshold=0.5, min_duration=0.05):
        """WAV 파일을 TAB으로 변환"""
        try:
            if not self.wav_converter:
                raise ValueError("WAV 변환기가 초기화되지 않았습니다. 모델 파일을 확인하세요.")
            
            # 임시 MIDI 파일 경로
            midi_path = wav_path.replace('.wav', '.mid')
            
            # WAV → MIDI 변환
            logger.info("WAV → MIDI 변환 중...")
            self.wav_converter.convert_wav_to_midi(
                wav_path=wav_path,
                output_path=midi_path,
                threshold=threshold,
                min_note_duration=min_duration
            )
            
            # MIDI → TAB 변환
            logger.info("MIDI → TAB 변환 중...")
            tab = self.tab_converter.convert_file(midi_path)
            
            if tab:
                # TAB 텍스트 생성
                tab_lines = tab.to_string()
                tab_content = f"Title: {tab.name}\n" + "=" * 50 + "\n\n"
                tab_content += "\n".join(tab_lines)
                
                # 임시 파일 정리
                try:
                    if os.path.exists(midi_path):
                        os.remove(midi_path)
                except:
                    pass
                
                return tab_content
            else:
                raise ValueError("TAB 변환에 실패했습니다.")
                
        except Exception as e:
            logger.error(f"변환 실패: {e}")
            raise
    
    def process_youtube_url(self, url, threshold=0.5, min_duration=0.05):
        temp_wav = None
        try:
            # 임시 파일 경로 생성
            temp_wav = os.path.join(TEMP_DIR, f"temp_audio_{os.getpid()}")
            
            # YouTube 다운로드
            self.download_youtube_audio(url, temp_wav)
            
            # 실제 생성된 파일 경로
            wav_file = temp_wav + ".wav"
            if not os.path.exists(wav_file):
                for ext in ['.wav', '.webm', '.m4a', '.mp3']:
                    test_path = temp_wav + ext
                    if os.path.exists(test_path):
                        wav_file = test_path
                        break
                else:
                    raise FileNotFoundError("다운로드된 오디오 파일을 찾을 수 없습니다.")
            
            # TAB 변환
            tab_content = self.convert_wav_to_tab(wav_file, threshold, min_duration)
            
            return tab_content
            
        finally:
            # 임시 파일 정리
            if temp_wav:
                for ext in ['.wav', '.webm', '.m4a', '.mp3', '']:
                    temp_file = temp_wav + ext
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except Exception:
                        pass

# API 인스턴스 생성
try:
    api = TabGeneratorAPI(MODEL_PATH)
    logger.info("API 서버 초기화 완료")
except Exception as e:
    logger.error(f"API 서버 초기화 실패: {e}")
    api = None

@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    status = "healthy" if api and api.wav_converter and api.tab_converter else "unhealthy"
    return jsonify({
        "status": status,
        "model_loaded": api.wav_converter is not None if api else False,
        "tab_converter_ready": api.tab_converter is not None if api else False
    })

@app.route('/convert_wav', methods=['POST'])
def convert_wav():
    """WAV 파일 업로드 및 변환"""
    if not api:
        return jsonify({"error": "API 서버가 초기화되지 않았습니다."}), 500
    
    try:
        # 파일 확인
        if 'file' not in request.files:
            return jsonify({"error": "파일이 업로드되지 않았습니다."}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "파일이 선택되지 않았습니다."}), 400
        
        # 매개변수 받기
        threshold = float(request.form.get('threshold', 0.5))
        min_duration = float(request.form.get('min_duration', 0.05))
        
        # 임시 파일 생성
        temp_wav = os.path.join(TEMP_DIR, f"upload_{os.getpid()}_{file.filename}")
        file.save(temp_wav)
        
        try:
            # TAB 변환
            logger.info(f"WAV 파일 변환 시작: {file.filename}")
            tab_content = api.convert_wav_to_tab(temp_wav, threshold, min_duration)
            
            return jsonify({
                "success": True,
                "tab_content": tab_content,
                "filename": file.filename
            })
            
        finally:
            # 임시 파일 정리
            try:
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
            except:
                pass
                
    except Exception as e:
        logger.error(f"WAV 변환 중 오류: {e}")
        return jsonify({"error": f"변환 중 오류가 발생했습니다: {str(e)}"}), 500

@app.route('/convert_youtube', methods=['POST'])
def convert_youtube():
    """YouTube URL 변환"""
    if not api:
        return jsonify({"error": "API 서버가 초기화되지 않았습니다."}), 500
    
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"error": "YouTube URL이 제공되지 않았습니다."}), 400
        
        url = data['url']
        threshold = data.get('threshold', 0.5)
        min_duration = data.get('min_duration', 0.05)
        
        # URL 유효성 검사
        if not any(domain in url for domain in ['youtube.com', 'youtu.be']):
            return jsonify({"error": "유효하지 않은 YouTube URL입니다."}), 400
        
        # TAB 변환
        logger.info(f"YouTube URL 변환 시작: {url}")
        tab_content = api.process_youtube_url(url, threshold, min_duration)
        
        return jsonify({
            "success": True,
            "tab_content": tab_content,
            "url": url
        })
        
    except Exception as e:
        logger.error(f"YouTube 변환 중 오류: {e}")
        return jsonify({"error": f"변환 중 오류가 발생했습니다: {str(e)}"}), 500

@app.route('/info', methods=['GET'])
def get_info():
    """API 정보"""
    return jsonify({
        "name": "기타 TAB 생성기 API",
        "version": "1.0.0",
        "description": "YouTube 링크 또는 WAV 파일을 기타 TAB 악보로 변환합니다.",
        "endpoints": {
            "/health": "서버 상태 확인",
            "/convert_wav": "WAV 파일 변환 (POST)",
            "/convert_youtube": "YouTube URL 변환 (POST)",
            "/info": "API 정보"
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "엔드포인트를 찾을 수 없습니다."}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "서버 내부 오류가 발생했습니다."}), 500

if __name__ == '__main__':
    # 모델 파일 경로 확인
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        print("\n" + "="*50)
        
    print(f"서버 주소: http://localhost:5000")
    
    # 자동 재시작 방지.
    app.run(host='0.0.0.0', port=5000, debug=False)