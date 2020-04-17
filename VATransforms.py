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

class ReduceAudioChannels:
    def __init__(self):
        pass
    
    def __call__(self, a):
        return torch.mean(a, dim=0, keepdim=True)

class NormalizeAudio:
    def __init__(self):
        pass
    
    def __call__(self, a):
        return a.div(a.abs().max().item())
