import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import librosa
import numpy as np
import pretty_midi
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')


class GuitarsetDataset(Dataset):
    def __init__(self, audio_dir, midi_dir, file_list, sr=22050, hop_length=512, segment_length=10):
        self.audio_dir = audio_dir
        self.midi_dir = midi_dir
        self.file_list = file_list
        self.sr = sr
        self.hop_length = hop_length
        self.segment_length = segment_length
        self.frame_length = segment_length * sr // hop_length

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        try:
            file_id = self.file_list[idx]

            # 오디오 데이터 로드
            audio_path = os.path.join(self.audio_dir, f"{file_id}_mic.wav")
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                return torch.zeros(128, self.frame_length), torch.zeros(88, self.frame_length)

            audio, _ = librosa.load(audio_path, sr=self.sr)

            # MIDI 데이터 로드
            midi_path = os.path.join(self.midi_dir, f"{file_id}.mid")
            if not os.path.exists(midi_path):
                print(f"MIDI file not found: {midi_path}")
                return torch.zeros(128, self.frame_length), torch.zeros(88, self.frame_length)

            midi_data = self.load_midi(midi_path)

            # 오디오와 MIDI 데이터를 동일한 길이의 세그먼트로 분할
            segments = self.create_segments(audio, midi_data)

            if not segments:
                print(f"No valid segments found for {file_id}")
                return torch.zeros(128, self.frame_length), torch.zeros(88, self.frame_length)

            # 랜덤하게 세그먼트 선택
            seg_idx = np.random.randint(len(segments))
            audio_seg, midi_seg = segments[seg_idx]

            return torch.FloatTensor(audio_seg), torch.FloatTensor(midi_seg)

        except Exception as e:
            print(f"Error loading data for index {idx}: {e}")
            return torch.zeros(128, self.frame_length), torch.zeros(88, self.frame_length)

    def load_midi(self, midi_path):
        """피아노롤로 변환 후 기타 음역대 설정 (E2-C7, MIDI note 40-128)"""
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            
            piano_roll = midi.get_piano_roll(fs=self.sr/self.hop_length)
            piano_roll = piano_roll[40:128, :]
            
            return piano_roll
        except Exception:
            return np.zeros((88, 1000))

    def create_segments(self, audio, midi_data):
        """오디오와 MIDI를 동일한 길이의 세그먼트로 분할"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, hop_length=self.hop_length, n_mels=128
        )
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        min_frames = min(mel_spec.shape[1], midi_data.shape[1])
        mel_spec = mel_spec[:, :min_frames]
        midi_data = midi_data[:, :min_frames]

        segments = []
        for start in range(0, min_frames - self.frame_length, self.frame_length // 2):
            end = start + self.frame_length
            if end <= min_frames:
                audio_seg = mel_spec[:, start:end]
                midi_seg = midi_data[:, start:end]
                segments.append((audio_seg, midi_seg))
        return segments


# 모델 아키텍처 정의
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


# 모델 학습 함수
def train_model(model, train_loader, val_loader, config, device='cuda'):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)

    # 스케줄러 설정
    warmup_epochs = 5
    main_epochs = config['num_epochs'] - warmup_epochs

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=1e-6)

    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    writer = SummaryWriter()

    for epoch in range(config['num_epochs']):
        # 훈련
        model.train()
        train_loss, num_batches = 0.0, 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Train]')

        for audio, midi in train_pbar:
            try:
                audio, midi = audio.to(device), midi.to(device)
                optimizer.zero_grad()
                output = model(audio)
                midi_binary = (midi > 0).float()
                loss = criterion(output, midi_binary)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1
                train_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'})

                global_step = epoch * len(train_loader) + num_batches
                writer.add_scalar('loss/train_loss_iter', loss.item(), global_step)

            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
        train_losses.append(avg_train_loss)
        writer.add_scalar('loss/train_loss_epoch', avg_train_loss, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # 검증
        model.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Val]')
            for audio, midi in val_pbar:
                try:
                    audio, midi = audio.to(device), midi.to(device)
                    output = model(audio)
                    midi_binary = (midi > 0).float()
                    loss = criterion(output, midi_binary)
                    val_loss += loss.item()
                    val_batches += 1
                    val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue

        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        val_losses.append(avg_val_loss)
        writer.add_scalar('loss/val_loss', avg_val_loss, epoch)

        # 스케줄러 업데이트
        scheduler.step()

        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')

        # 모델 저장
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"models/checkpoint_epoch_{epoch+1}.pth")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "models/best_model.pth")
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")

    writer.close()
    return train_losses, val_losses


# MIDI 예측 및 저장 함수
def predict_and_save_midi(model, audio_path, output_path, device='cuda', threshold=0.5):
    model.eval()
    model.to(device)
    audio, sr = librosa.load(audio_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=512, n_mels=128)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    segment_frames = 10 * sr // 512
    hop_frames = segment_frames // 2
    predictions = []
    
    with torch.no_grad():
        for start in range(0, mel_spec.shape[1] - segment_frames, hop_frames):
            end = start + segment_frames
            segment = mel_spec[:, start:end]
            segment_tensor = torch.FloatTensor(segment).unsqueeze(0).to(device)
            pred = model(segment_tensor)
            predictions.append(pred.cpu().numpy()[0])
            
    full_prediction = np.zeros((88, mel_spec.shape[1]))
    frame_counts = np.zeros((88, mel_spec.shape[1]))
    
    for i, pred in enumerate(predictions):
        start = i * hop_frames
        end = start + segment_frames
        full_prediction[:, start:end] += pred
        frame_counts[:, start:end] += 1
        
    full_prediction /= frame_counts
    create_midi_from_prediction(full_prediction, output_path, threshold=threshold, fs=sr/512)

def create_midi_from_prediction(prediction, output_path, threshold=0.5, fs=43.0664):
    midi = pretty_midi.PrettyMIDI()
    guitar = pretty_midi.Instrument(program=24)
    binary_pred = prediction > threshold

    for note_idx in range(88):
        note_num = note_idx + 40
        active_frames = np.where(binary_pred[note_idx])[0]
        if not len(active_frames): continue

        note_starts, note_ends = [], []
        start_frame = active_frames[0]
        for i in range(1, len(active_frames)):
            if active_frames[i] != active_frames[i-1] + 1:
                note_starts.append(start_frame / fs)
                note_ends.append(active_frames[i-1] / fs)
                start_frame = active_frames[i]
        note_starts.append(start_frame / fs)
        note_ends.append(active_frames[-1] / fs)

        for start, end in zip(note_starts, note_ends):
            if end - start > 0.05:
                note = pretty_midi.Note(velocity=100, pitch=note_num, start=start, end=end)
                guitar.notes.append(note)

    midi.instruments.append(guitar)
    midi.write(output_path)


def main():
    os.makedirs("models", exist_ok=True)
    
    config = {
        "num_epochs": 100,
        "batch_size": 32,         
        "learning_rate": 0.001,
        "num_workers": 8          
    }

    audio_dir = "Guitarset/audio"
    midi_dir = "Guitarset/midi"

    if not os.path.exists(audio_dir) or not os.path.exists(midi_dir):
        print(f"Dataset directory not found.")
        return

    audio_files = [f.replace('_mic.wav', '') for f in os.listdir(audio_dir) if f.endswith('_mic.wav')]
    if len(audio_files) < 2:
        print("Not enough files for train/validation split.")
        return

    train_files, val_files = train_test_split(audio_files, test_size=0.2, random_state=42)
    print(f"Training files: {len(train_files)}, Validation files: {len(val_files)}")

    print("Creating datasets and dataloaders...")
    train_dataset = GuitarsetDataset(audio_dir, midi_dir, train_files)
    val_dataset = GuitarsetDataset(audio_dir, midi_dir, val_files)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)

    print("Creating model...")
    model = GuitarTranscriptionModel()
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Starting training...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, config, device)

    torch.save(model.state_dict(), 'models/guitar_transcription_final.pth')
    print("Final model saved.")

    if val_files:
        print("Running prediction on a sample file...")
        model.load_state_dict(torch.load("models/best_model.pth"))
        test_audio = os.path.join(audio_dir, f"{val_files[0]}_mic.wav")
        predict_and_save_midi(model, test_audio, 'models/predicted_output.mid', device=device)
        print("Test prediction saved as 'predicted_output.mid'")

if __name__ == "__main__":
    main()