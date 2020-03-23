from torch.utils.data import Dataset
import constants
import torchvision.io as io
import os
import torchaudio.transforms as AT
import torchvision.transforms as IT
import json
import torch

class VideoTransform:
    def __init__(self, f):
        self.f = f

    def __call__(self, v):
        ans = []
        for img in v:
            out = self.f(img)
            if torch.is_tensor(out):
                out = torch.unsqueeze(out, dim=0)
            
            ans += [out]

        return torch.cat(ans, dim=0) if torch.is_tensor(ans[0]) else ans

class ReduceAudioChannels():
    def __init__(self):
        pass
    
    def __call__(self, a):
        return torch.mean(a, dim=0, keepdim=True)

class NormalizeAudio:
    def __init__(self):
        pass
    
    def __call__(self, a):
        return a.div(a.abs().max().item())

class CoughDataset(Dataset):
    def __init__(self, root_dir = constants.DATA_BASE_DIR, chunk_size = constants.CHUNK_SIZE):
        fs = [f for f in os.listdir(root_dir) if f.endswith('op_rs.mp4')]
        self.data = []
        labels = json.loads(open(os.path.join(root_dir, 'labels.json'), 'r').read())

        self.audio_transforms = IT.Compose([
            ReduceAudioChannels(),
            NormalizeAudio(),
            AT.Resample(constants.AUDIO_SAMPLE_RATE, constants.RESAMPLED_AUDIO_SAMPLE_RATE),
            AT.MelSpectrogram(sample_rate=constants.RESAMPLED_AUDIO_SAMPLE_RATE)
        ])

        self.video_transforms = IT.Compose([
            VideoTransform(IT.ToPILImage()),
            VideoTransform(IT.Resize((constants.INPUT_FRAME_WIDTH, constants.INPUT_FRAME_WIDTH))),
            VideoTransform(IT.ToTensor()),
            VideoTransform(IT.Normalize(mean=constants.MEAN, std=constants.STD)),
        ])

        for f in fs:
            #break in 1 sec chunks and add label
            chunks = self.break_in_chunks(os.path.join(root_dir, f), os.path.join(root_dir, f[:-10] + '_rs.mp4'),  labels[f], chunk_size)
            self.data += chunks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]

    def break_in_chunks(self, video_file, audio_file, cough_times, chunk_size):
        v = io.read_video(video_file, pts_unit='sec')[0]
        a = io.read_video(audio_file, pts_unit='sec')[1]

        v = v.permute([0, 3, 1, 2])

        ans = []
        #break into chunks
        end_frame = int(v.shape[0] / constants.VIDEO_FPS) * constants.VIDEO_FPS
        vid_range = range(0, end_frame, constants.VIDEO_FPS)
        
        end_audio_frame = int(a.shape[1] / constants.AUDIO_SAMPLE_RATE) * constants.AUDIO_SAMPLE_RATE
        audio_range = range(0, end_audio_frame, constants.AUDIO_SAMPLE_RATE)

        for i, (v_frame, a_frame) in enumerate(zip(vid_range, audio_range)): 
            #apply transforms
            v_chunk = self.video_transforms(v[v_frame:v_frame + constants.VIDEO_FPS])
            a_chunk = self.audio_transforms(a[:, a_frame:a_frame + constants.AUDIO_SAMPLE_RATE])
            
            ans.append(
                (v_chunk.permute([1, 0, 2, 3]), a_chunk, 1 if i in cough_times else 0)
            )

        return ans