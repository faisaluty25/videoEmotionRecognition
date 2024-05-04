from src.data import get_data_loaders
from src.train import optimize
from src.helpers import load_model, replace_insatance
import torch.nn.functional as F
import src.model as models
from torch.optim import lr_scheduler
import torch

print_model=1
batch_size = 512  # size of the minibatch for stochastic gradient descent (or Adam)
num_epochs = 70     # number of epochs for training
num_classes = 1000       # number of classes. Do not change this
learning_rate =0.05*(0.6)**0  # Learning rate for SGD (or Adam)
weight_decay = 0.001     # regularization. Increase this to combat overfitting
momentum=0.9 
accumulation_steps=2
model_name= "SRegions24_Prelu24g3k5_200_Global"
#suffix='plateau' +'b512_230siz_triv_0.05s'+'0.45s'+'0.02s'+'0.01s'
#suffix='200'+'220'+'topk'
suffix=''
# Note: for suffix='pytorch_ric' lr=0.05*(0.6)**5
data_loaders = get_data_loaders(batch_size=batch_size)
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True

#x=torch.rand(64,7,7,90).flatten(start_dim=1, end_dim=2)
#_,idx=x.topk(k=3, dim=1, sorted=False)
#print(x.shape, idx.shape, x[idx].shape)
#torch.mean(_, dim=1).shape
# instance model MyModel with num_classes and drouput defined in the previous
# cell
model = getattr(models, model_name)(num_classes=num_classes)
# initialize
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.orthogonal_(m.weight)

model.apply(init_weights)
model.post=models.Topk(k=5)
# Get the optimizer using get_optimizer and the model you just created, the learning rate,
# the optimizer and the weight decay specified in the previous cel
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum,weight_decay=weight_decay,)


#milestones=[5,9,12, *range(12,num_epochs,2)]
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=2,threshold=1e-3,verbose=True)
#scheduler = lr_scheduler.ExponentialLR(optimizer,gamma=0.7)

def step(loss ,epoch=None):
    scheduler.step(loss)

def loss(output,target):
    #target=F.one_hot(target, num_classes).float()
    return F.cross_entropy(output, target,label_smoothing=0.03)




s_epoch=load_model(model_name+suffix,model)

#for p in model.parameters():
#    requires_grad=False

#model.Dconvs0[0]=models.SSepConv(3,32,3,stride=2 ,padding=1,growth_factor=3 ,depth_count=3)
#model.Dconvs0[1]=models.SSepConv(32,42,3,stride=(2,1) ,padding=1,growth_factor=3 ,depth_count=3)
#model.Dconvs0[2]=models.SSepConv(42,54,3,stride=(1,2) ,padding=1,growth_factor=3 ,depth_count=3)


if print_model:
    print(f"model {model_name} has :{sum(p.numel() for p in model.parameters())/1e6} M parameters ")
    print(f"Effictive W>0.01 precentage: ")
    print('\n'.join('layer {} has : {}'.format(n,torch.sum(torch.abs(p)>0.01)/p.numel()) for n, p in model.named_parameters()))
#replace_insatance(model,torch.nn.LeakyReLU, models.Swish())
#print(model)

optimize(
    data_loaders,
    model,
    optimizer,
    loss,
    s_epoch=s_epoch,
    n_epochs=num_epochs,
    model_name=model_name+suffix,
    step=step,
    accumulation_steps=accumulation_steps,
    run_logs=True,
    
)


