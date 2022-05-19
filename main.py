import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
from datetime import datetime

from model import Generator,Discriminator,ADLClassifier
from GE2E_loss import GE2ELoss
from pre_processing import *
import gc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
context_size=13
#data loader
dataset = ADLDataset()
loader = DataLoader(dataset,batch_size=batch_size, shuffle=False, num_workers=num_workers,drop_last=True) 

#Model and optimizers
G = Generator(context_size)
D = Discriminator()
C = ADLClassifier(context_size)
G.to(device)
D.to(device)
C.to(device)

g_optimizer = torch.optim.Adam(G.parameters(), lr, [0.5, 0.999])
d_optimizer = torch.optim.Adam(D.parameters(), lr, [0.5, 0.999])
c_optimizer = torch.optim.Adam(C.parameters(), lr, [0.5, 0.999])
scheduler_g = StepLR(g_optimizer, step_size=10, gamma=1-decay)
scheduler_d = StepLR(d_optimizer, step_size=10, gamma=1-decay)
scheduler_c = StepLR(c_optimizer, step_size=10, gamma=1-decay)

#classifier loss
CELoss = nn.CrossEntropyLoss()
cos = nn.CosineSimilarity()
#loading speaker embedding model
if sys.argv[1]=='subject_transfer': 
    dvector = torch.jit.load(checkpoint_path).eval().to(device)
    GE2E_loss = GE2ELoss().to(device)
    GE2E_optimizer = torch.optim.Adam(GE2E_loss.parameters(), 1e-3, [0.5, 0.999])

def d_vector(train_iter,n_speakers):
        batch = train_iter
        batch = batch.reshape(n_speakers,19,300)
        batch = torch.swapaxes(batch, 1,2).to(device)
        emb = torch.zeros(n_speakers,64)
        with torch.no_grad():
            for i in range(n_speakers):
                emb[i] = dvector.embed_utterance(batch[i]).detach()
        return emb.detach()
def gradient_penalty(y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)
# learning
data_iter = iter(loader)
start_time = datetime.now()
for i in range(num_iters):
    try:
        x_real, speaker_idx_org, context_idx_org, label_org, normalizer = next(data_iter)
    except:
        data_iter = iter(loader)
        x_real, speaker_idx_org, context_idx_org, label_org, normalizer = next(data_iter)           

    #D-vector
    if sys.argv[1]=='subject_transfer':
        label_org = d_vector(x_real,len(x_real))
    #randomly select target subject or context
    rand_idx = torch.randperm(label_org.size(0))
    label_trg = label_org[rand_idx]
    context_idx_trg = context_idx_org[rand_idx]
    #move to cuda
    x_real,label_org,label_trg = x_real.to(device),label_org.to(device),label_trg.to(device) 
    context_idx_trg,context_idx_org = context_idx_trg.to(device),context_idx_org.to(device)

    #classify real data

    cls_real = C(x_real)
    if sys.argv[1]=='subject_transfer':
        cls_loss_real = (1-cos(cls_real,label_org)) + GE2E_loss(cls_real)
    else:
        cls_loss_real = CELoss(input=cls_real, target=context_idx_org) 


    #update discriminators
    out_r = D(x_real)
    x_fake = G(x_real, label_trg)

    out_f = D(x_fake.detach())
    d_loss_t = F.binary_cross_entropy(input=out_f,target=torch.zeros_like(out_f, dtype=torch.float)) + \
        F.binary_cross_entropy(input=out_r, target=torch.ones_like(out_r, dtype=torch.float))
    
    # Compute loss for gradient penalty.
    alpha = torch.rand(x_real.size(0), 1, 1).to(device)
    x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
    out_src = D(x_hat)
    d_loss_gp = gradient_penalty(out_src, x_hat)

    d_loss = d_loss_t + lambda_cls * cls_loss_real + d_loss_gp * lambda_gp

    g_optimizer.zero_grad()
    d_optimizer.zero_grad()
    c_optimizer.zero_grad()
    if sys.argv[1]=='subject_transfer':
        GE2E_optimizer.zero_grad()
    d_loss.backward()
    c_optimizer.step()
    d_optimizer.step()

    loss = {}
    loss['C/C_loss'] = cls_loss_real.item()
    loss['D/D_loss'] = d_loss.item()

    #update generator
    if i % n_critic == 0 and i>0:
        x_fake = G(x_real, label_trg)
        g_out_src = D(x_fake)
        g_loss_fake = F.binary_cross_entropy_with_logits(input=g_out_src, target=torch.ones_like(g_out_src, dtype=torch.float))
        
        out_cls = C(x_fake)

        if sys.argv[1]=='subject_transfer':
            g_loss_cls = (1-cos(out_cls,label_trg)) + GE2E_loss(out_cls)
        else:
            g_loss_cls = CELoss(input=out_cls, target=context_idx_trg) 

        # Target-to-original domain.
        x_reconst = G(x_fake, label_org)
        g_loss_rec = F.l1_loss(x_reconst, x_real )

        # Original-to-Original domain(identity).
        # x_fake_iden = G(x_real, label_org)
        # id_loss = F.l1_loss(x_fake_iden, x_real )

        # Backward and optimize.
        g_loss = g_loss_fake + lambda_rec * g_loss_rec + lambda_cls * g_loss_cls# + lambda_identity * id_loss
         
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        c_optimizer.zero_grad()
        if sys.argv[1]=='subject_transfer':
            GE2E_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        scheduler_g.step()
        scheduler_d.step()
        scheduler_c.step()
        # Logging.
        loss['G/loss_fake'] = g_loss_fake.item()
        loss['G/loss_rec'] = g_loss_rec.item()
        loss['G/loss_cls'] = g_loss_cls.item()
        loss['G/loss_id'] = id_loss.item()
        loss['G/g_loss'] = g_loss.item()

        #logging
        if i % log_step == 0 and i>0:
            t = datetime.now() - start_time
            log = "Elapsed [{}], Iteration [{}/{}]".format(str(t)[:-7], i, num_iters)
            for tag, value in loss.items():
                log += ", {}: {:.4f}".format(tag, value)
            print(log)
            for tag, value in loss.items():
                logger.scalar_summary(tag, value, i)

        if i % model_save_step == 0 and i>0:
            G_path = os.path.join(model_save_dir, '{}-G.ckpt'.format(i+1))
            D_path = os.path.join(model_save_dir, '{}-D.ckpt'.format(i+1))
            C_path = os.path.join(model_save_dir, '{}-C.ckpt'.format(i+1))
            torch.save(G.state_dict(), G_path)
            torch.save(D.state_dict(), D_path)
            torch.save(C.state_dict(), C_path)
            print('Saving checkpoint...')
            gc.collect()

