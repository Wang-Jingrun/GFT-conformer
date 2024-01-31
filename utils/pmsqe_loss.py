import torch, os 
from asteroid_filterbanks import STFTFB, Encoder, transforms
from asteroid.losses import PITLossWrapper, SingleSrcPMSQE


class pmsqe_loss_fn:
    def __init__(self, config):
        self.device = torch.device(config['device'])
        self.pmsqe_stft = Encoder(
            STFTFB(kernel_size=512, n_filters=512, stride=256)).to(self.device)
        self.pmsqe_loss = PITLossWrapper(SingleSrcPMSQE(), pit_from='pw_pt').to(self.device)

    def __call__(self, est_wav, ref_wav):
        ref_wav = ref_wav.reshape(-1, 2, 16000)
        est_wav = est_wav.reshape(-1, 2, 16000)

        ref_spec = transforms.mag(self.pmsqe_stft(ref_wav))
        est_spec = transforms.mag(self.pmsqe_stft(est_wav))

        p_loss = self.pmsqe_loss(est_spec, ref_spec)
        return p_loss