import lightning as pl

class DLVAE(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(DLVAE, self).__init__(*args, **kwargs)
        
        