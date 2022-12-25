import torch 
import numpy as np

from utils.utils import make_missing

class GraphSamplerTrainer: 

    def __init__(self, print_every:int = 5): 
        super().__init__() 
        self.print_every = print_every

    def __call__(self, args, model, 
                optimizer, 
                device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        model.train()
        for epoch in range(args.pretrain_epoch): 
            # training loop
            with torch.set_grad_enabled(True): 
                tr_loss = getattr(model, 'graph_sampling').train_step()['total_loss']
                model.zero_grad()
                optimizer.zero_grad()
                tr_loss.backward()
                optimizer.step() 
        
            if epoch % self.print_every == 0:
                prefix = f'Epoch [{epoch+1}/{args.epoch}]'
                print(f'{prefix}: training loss= {tr_loss.detach().cpu().item():.6f}')
            