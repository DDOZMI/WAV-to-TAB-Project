import os
import torch
import torch.nn as nn
import librosa
import numpy as np
import pretty_midi
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class GuitarTranscriptionModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, num_layers=2, output_dim=88):
        super(GuitarTranscriptionModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        cnn_output_dim = 256 * (input_dim // 4)
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(hidden_dim, output_dim), nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, _, time_frames = x.shape
        x = x.unsqueeze(1)
        x = self.cnn(x)
        _, channels, height, width = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch_size, width, channels * height)
        lstm_out, _ = self.lstm(x)
        output = self.output_layer(lstm_out)
        return output.permute(0, 2, 1)


class WAVToMIDIConverter:
    def __init__(self, model_path, device='cuda', sr=22050, hop_length=512):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.sr = sr
        self.hop_length = hop_length
        self.segment_length = 10  # 초 단위
        self.frame_length = self.segment_length * sr // hop_length
        
        # 모델 로드
        self.model = self._load_model(model_path)
        print(f"모델이 {self.device}에서 로드되었습니다.")
    
    def _load_model(self, model_path):
        """학습된 모델 로드"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        model = GuitarTranscriptionModel()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def convert_wav_to_midi(self, wav_path, output_path, threshold=0.5, min_note_duration=0.05):
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"WAV 파일을 찾을 수 없습니다: {wav_path}")
        
        print(f"WAV 파일 로드 중: {wav_path}")
        
        # 오디오 로드 및 전처리
        audio, _ = librosa.load(wav_path, sr=self.sr)
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, hop_length=self.hop_length, n_mels=128
        )
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        print(f"오디오 길이: {len(audio)/self.sr:.2f}초")
        print(f"Mel spectrogram 크기: {mel_spec.shape}")
        
        # 세그먼트별 예측
        predictions = self._predict_segments(mel_spec)
        
        # 전체 예측 결과 합성
        full_prediction = self._merge_predictions(predictions, mel_spec.shape[1])
        
        # MIDI 생성
        self._create_midi_file(full_prediction, output_path, threshold, min_note_duration)
        
        print(f"MIDI 파일 저장 완료: {output_path}")
    
    def _predict_segments(self, mel_spec):
        hop_frames = self.frame_length // 2
        predictions = []
        
        print("음표 예측 중...")
        
        # 세그먼트 생성 및 예측
        segments_count = max(1, (mel_spec.shape[1] - self.frame_length) // hop_frames + 1)
        
        with torch.no_grad():
            for start in tqdm(range(0, mel_spec.shape[1] - self.frame_length + 1, hop_frames), 
                            desc="세그먼트 처리"):
                end = start + self.frame_length
                
                # 세그먼트 추출
                segment = mel_spec[:, start:end]
                
                # 길이가 부족한 경우 패딩
                if segment.shape[1] < self.frame_length:
                    padding = self.frame_length - segment.shape[1]
                    segment = np.pad(segment, ((0, 0), (0, padding)), mode='constant')
                
                # 텐서로 변환 및 예측
                segment_tensor = torch.FloatTensor(segment).unsqueeze(0).to(self.device)
                pred = self.model(segment_tensor)
                predictions.append(pred.cpu().numpy()[0])
        
        return predictions
    
    def _merge_predictions(self, predictions, total_frames):
        hop_frames = self.frame_length // 2
        full_prediction = np.zeros((88, total_frames))
        frame_counts = np.zeros((88, total_frames))
        
        print("예측 결과 합성 중...")
        
        for i, pred in enumerate(predictions):
            start = i * hop_frames
            end = min(start + self.frame_length, total_frames)
            actual_length = end - start
            
            full_prediction[:, start:end] += pred[:, :actual_length]
            frame_counts[:, start:end] += 1
        
        # 겹치는 부분의 평균 계산
        frame_counts[frame_counts == 0] = 1  # 0으로 나누는 것 방지
        full_prediction /= frame_counts
        
        return full_prediction
    
    def _create_midi_file(self, prediction, output_path, threshold, min_note_duration):

        fs = self.sr / self.hop_length  # 프레임당 시간
        
        midi = pretty_midi.PrettyMIDI()
        guitar = pretty_midi.Instrument(program=24)  # 어쿠스틱 기타
        
        binary_pred = prediction > threshold
        note_count = 0
        
        print("MIDI 음표 생성 중...")
        
        for note_idx in tqdm(range(88), desc="음표 처리"):
            note_num = note_idx + 40  # MIDI 음표 번호 (40 = E2)
            active_frames = np.where(binary_pred[note_idx])[0]
            
            if len(active_frames) == 0:
                continue
            
            # 연속된 프레임들을 음표로 그룹화
            note_starts, note_ends = [], []
            start_frame = active_frames[0]
            
            for i in range(1, len(active_frames)):
                if active_frames[i] != active_frames[i-1] + 1:  # 불연속 지점
                    note_starts.append(start_frame / fs)
                    note_ends.append(active_frames[i-1] / fs)
                    start_frame = active_frames[i]
            
            # 마지막 음표 추가
            note_starts.append(start_frame / fs)
            note_ends.append(active_frames[-1] / fs)
            
            # 음표 생성 및 추가
            for start_time, end_time in zip(note_starts, note_ends):
                duration = end_time - start_time
                if duration >= min_note_duration:  # 최소 길이 필터링
                    note = pretty_midi.Note(
                        velocity=100, 
                        pitch=note_num, 
                        start=start_time, 
                        end=end_time
                    )
                    guitar.notes.append(note)
                    note_count += 1
        
        midi.instruments.append(guitar)
        midi.write(output_path)
        
        print(f"총 {note_count}개의 음표가 생성되었습니다.")
    
    def batch_convert(self, input_dir, output_dir, threshold=0.5, min_note_duration=0.05):
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"입력 디렉토리를 찾을 수 없습니다: {input_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
        
        if not wav_files:
            print("WAV 파일을 찾을 수 없습니다.")
            return
        
        print(f"{len(wav_files)}개의 WAV 파일을 변환합니다...")
        
        for wav_file in tqdm(wav_files, desc="배치 변환"):
            wav_path = os.path.join(input_dir, wav_file)
            midi_file = wav_file.replace('.wav', '.mid')
            midi_path = os.path.join(output_dir, midi_file)
            
            try:
                self.convert_wav_to_midi(wav_path, midi_path, threshold, min_note_duration)
            except Exception as e:
                print(f"변환 실패 {wav_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description='WAV 파일을 MIDI 파일로 변환')
    parser.add_argument('--model', required=True, help='학습된 모델 파일 경로 (.pth)')
    parser.add_argument('--input', required=True, help='입력 WAV 파일 또는 디렉토리 경로')
    parser.add_argument('--output', required=True, help='출력 MIDI 파일 또는 디렉토리 경로')
    parser.add_argument('--threshold', type=float, default=0.5, help='음표 검출 임계값 (기본값: 0.5)')
    parser.add_argument('--min_duration', type=float, default=0.05, help='최소 음표 길이 in 초 (기본값: 0.05)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='사용할 디바이스')
    parser.add_argument('--batch', action='store_true', help='디렉토리 내 모든 WAV 파일 변환')
    
    args = parser.parse_args()
    
    # 변환기 초기화
    converter = WAVToMIDIConverter(args.model, device=args.device)
    
    if args.batch:
        # 배치 변환
        converter.batch_convert(
            input_dir=args.input, 
            output_dir=args.output, 
            threshold=args.threshold, 
            min_note_duration=args.min_duration
        )
    else:
        # 단일 파일 변환
        converter.convert_wav_to_midi(
            wav_path=args.input, 
            output_path=args.output, 
            threshold=args.threshold, 
            min_note_duration=args.min_duration
        )

if __name__ == "__main__":
    # 사용법
    if len(os.sys.argv) == 1:
        print("=== WAV to MIDI 사용법 ===")
        print()
        print("단일 파일 변환:")
        print("python wav_to_midi.py --model best_model.pth --input input.wav --output output.mid")
        print()
        print("일괄 변환:")
        print("python wav_to_midi.py --model best_model.pth --input ./wav_files/ --output ./midi_files/ --batch")
        print()
        print("추가 옵션:")
        print("--threshold 0.3     # 임계값 조정 (낮을수록 더 많은 음표 검출)")
        print("--min_duration 0.1  # 최소 음표 길이 조정")
        print("--device cpu        # CPU 사용")
        print()
        
    else:
        main()