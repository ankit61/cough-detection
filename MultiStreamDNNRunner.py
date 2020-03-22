from MultiStreamDNN import MultiStreamDNN
from BaseResNetRunner import BaseResNetRunner

class MultiStreamDNNRunner(BaseResNetRunner):
    def __init__(self, load_paths=None):
        net = MultiStreamDNN()
        super(MultiStreamDNNRunner, self).__init__(net, load_paths=load_paths)

    def do_forward_pass(self, batch):
        return self.nets[0](batch[0], batch[1]).squeeze(dim=1)