from torch import nn

class Encoder(nn.Module):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def forward(self,X,*args):
        raise NotImplementedError
    
class Decoder(nn.Module):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def init_state(self,enc_outputs,*args):
        raise NotImplementedError
    
    def forward(self,X,state):
        raise NotImplementedError
    
class Encoder_Decoder(nn.Module):
    def __init__(self,encoder,decoder,**kwargs):
        super().__init__(**kwargs)
        self.encoder=encoder
        self.decoder=decoder
    
    def forward(self,enc_X,dec_X,*args):
        enc_outputs=self.encoder(enc_X,*args)
        dec_state=self.decoder.init_state(enc_outputs,*args)
        return self.decoder(dec_X,dec_state)