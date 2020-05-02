from torch.utils.data import Dataset
import constants
import torchvision.io as io
import os
import torchaudio.transforms as AT
import torchvision.transforms as IT
import json
import torch
from VATransforms import VideoTransform, ReduceAudioChannels, NormalizeAudio


class CoughDataset(Dataset):
    def __init__(self, root_dir=constants.DATA_BASE_DIR, result_mode=False, chunk_size=constants.CHUNK_SIZE, model_type='all'):

        assert chunk_size == 1, 'current implementation only supports 1 second chunks'

        fs = [f for f in os.listdir(root_dir) if f.endswith(constants.VISUAL_SUFFIX)]
        self.data = []
        self.meta = []
        if not result_mode:
            labels = json.loads(open(os.path.join(root_dir, 'labels.json'), 'r').read())

        self.ensemble_audio_transforms = [
            IT.Compose([
                ReduceAudioChannels(),
                NormalizeAudio(),
                AT.Resample(constants.AUDIO_SAMPLE_RATE, constants.RESAMPLED_AUDIO_SAMPLE_RATE),
                AT.MFCC(sample_rate=constants.RESAMPLED_AUDIO_SAMPLE_RATE, n_mfcc=constants.N_MFCCS)
            ]),
            IT.Compose([
                ReduceAudioChannels(),
                NormalizeAudio(),
                AT.Resample(constants.AUDIO_SAMPLE_RATE, constants.RESAMPLED_AUDIO_SAMPLE_RATE),
                AT.MFCC(sample_rate=constants.RESAMPLED_AUDIO_SAMPLE_RATE, n_mfcc=constants.N_MFCCS)
            ])
        ]

        self.ensemble_video_transforms = [
            IT.Compose([
                VideoTransform(IT.ToPILImage()),
                VideoTransform(IT.Resize((constants.INPUT_FRAME_WIDTH, constants.INPUT_FRAME_WIDTH))),
                VideoTransform(IT.ToTensor()),
                VideoTransform(IT.Normalize(mean=constants.MEAN, std=constants.STD)),
            ]),
            IT.Compose([
                VideoTransform(IT.ToPILImage()),
                VideoTransform(IT.Resize((constants.INPUT_FRAME_WIDTH, constants.INPUT_FRAME_WIDTH))),
                VideoTransform(IT.ToTensor()),
                VideoTransform(IT.Normalize(mean=constants.MEAN, std=constants.STD)),
            ])
        ]

        self.ensemble_video_post_transforms = [
            lambda x: x.permute([1, 0, 2, 3]),
            lambda x: x.permute([1, 0, 2, 3])
        ]

        self.ensemble_audio_post_transforms = [
            lambda x: x,
            lambda x: x
        ]

        if model_type == 'conv3D_MFCCs':
            self.ensemble_video_transforms = [self.ensemble_video_transforms[0]]
            self.ensemble_audio_transforms = [self.ensemble_audio_transforms[0]]
            self.ensemble_video_post_transforms = [self.ensemble_video_post_transforms[0]]
            self.ensemble_audio_post_transforms = [self.ensemble_audio_post_transforms[0]]

        for f in fs:
            # break in 1 sec chunks and add label
            audio_file = f[:-len(constants.VISUAL_SUFFIX)] + constants.AUDIO_SUFFIX
            chunks, meta = self.break_in_chunks(os.path.join(root_dir, f), os.path.join(root_dir, audio_file),  [] if result_mode else labels[f], chunk_size)
            self.data += chunks
            self.meta += meta

        if not result_mode:
            self.print_data_stats()

    def print_data_stats(self):
        print('Printing data statistics...')
        print('Positive Label Rate:', sum([data_tuple[-1] for data_tuple in self.data]) / len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]

    def get_meta(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.meta[idx]

    def break_in_chunks(self, video_file, audio_file, cough_times, chunk_size):
        v = io.read_video(video_file, pts_unit='sec')[0]
        a = io.read_video(audio_file, pts_unit='sec')[1]

        v = v.permute([0, 3, 1, 2])

        ans = []
        meta = []
        # break into chunks
        end_frame = int(v.shape[0] / constants.VIDEO_FPS) * constants.VIDEO_FPS
        vid_range = range(0, end_frame, constants.VIDEO_FPS)

        end_audio_frame = int(a.shape[1] / constants.AUDIO_SAMPLE_RATE) * constants.AUDIO_SAMPLE_RATE
        audio_range = range(0, end_audio_frame, constants.AUDIO_SAMPLE_RATE)

        for i, (v_frame, a_frame) in enumerate(zip(vid_range, audio_range)):
            # apply transforms
            v_chunk = v[v_frame:v_frame + constants.VIDEO_FPS]
            a_chunk = a[:, a_frame:a_frame + constants.AUDIO_SAMPLE_RATE]

            cur_ans = ()

            for j in range(len(self.ensemble_video_transforms)):
                cur_ans += (self.ensemble_video_post_transforms[j](
                                self.ensemble_video_transforms[j](v_chunk)
                            ),)

                cur_ans += (self.ensemble_audio_post_transforms[j](
                                self.ensemble_audio_transforms[j](a_chunk)
                            ),)

            cur_ans += (1 if i in cough_times else 0, )

            ans.append(cur_ans)

            meta.append((os.path.basename(video_file), [i, i + 1]))

        return ans, meta
