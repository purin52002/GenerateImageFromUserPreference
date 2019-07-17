class FeatureNumber:
    def __init__(self, fmap_base=4096, fmap_decay=1.0, fmap_max=256):
        self.fmap_base = fmap_base
        self.fmap_decay = fmap_decay
        self.fmap_max = fmap_max

    def __call__(self, stage: int):
        return min(
            int(self.fmap_base / (2.0 ** (stage * self.fmap_decay))),
            self.fmap_max)