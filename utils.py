import h5py
import torch
import shutil

def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
            
def save_checkpoint(state, is_best,task_id, filename='checkpoint.pth.tar'):
    torch.save(state, task_id+filename)
    if is_best:
        shutil.copyfile(task_id+filename, task_id+'model_best.pth.tar')

def auto_loss(snet_out, cls_err, variables, M, alpha):
    w = None
    for p in variables:
        w = torch.cat((w, p.view(-1))) if w is not None else p.view(-1)
    l1 = F.l1_loss(w, torch.zeros_like(w))
    loss = 1-snet_out
    # loss= loss*torch.log(torch.clamp(cls_err+1e-5, min=1e-5, max=1))
    loss = loss * cls_err
    # print(loss)
    # loss = loss + snet_out*torch.log(torch.clamp(1-cls_err, min=1e-5, max=1))
    loss = loss + snet_out * torch.clamp(M-cls_err, min=0)
    # print(loss)
    # loss = -1 * loss
    res = torch.sum(loss)+alpha*l1
    return res

def mse_loss(output, target):
    loss = torch.pow((output - target), 2)
    return loss