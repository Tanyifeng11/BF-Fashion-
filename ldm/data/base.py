from torch.utils.data import IterableDataset


class Txt2ImgIterableBaseDataset(IterableDataset):
    """Compatibility base class used by the original latent-diffusion dataloader."""

    def __init__(self, num_records=0, valid_ids=None, size=256):
        super().__init__()
        self.num_records = num_records
        self.valid_ids = valid_ids if valid_ids is not None else list(range(num_records))
        self.sample_ids = self.valid_ids
        self.size = size

    def __len__(self):
        return self.num_records
