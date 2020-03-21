import torch
import torch.nn as nn
import tensorboardX
from abc import abstractmethod, ABCMeta
import constants

#torch.set_default_tensor_type(torch.cuda.FloatTensor 
#    if torch.cuda.is_available() else torch.FloatTensor)

class BaseModule(nn.Module):
    __metaclass__ = ABCMeta
    def __init__(self, debug = True):
        super(BaseModule, self).__init__()
        self.debug  = debug
        self.writer = tensorboardX.SummaryWriter(constants.TENSORBOARDX_BASE_DIR)

    @abstractmethod
    def introspect(self):
        return
