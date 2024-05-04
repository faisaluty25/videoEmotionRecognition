from src.data import get_data_loaders
from src.train import valid_one_epoch
from src.helpers import load_model, replace_insatance
import torch.nn.functional as F
import src.model as models
import torch
from torchvision.models import shufflenet_v2_x1_0

batch_size = 512  # size of the minibatch for stochastic gradient descent (or Adam)
num_classes = 1000       # number of classes. Do not change this
model_name= "SRegions24_Prelu32_572x2_Global"
#suffix='plateau' +'b512_230siz_triv_0.05s'+'0.45s'+'0.02s'+'0.01s'
suffix=''
# Note: for suffix='pytorch_ric' lr=0.05*(0.6)**5
data_loaders = get_data_loaders(batch_size=batch_size)
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True

# instance model MyModel with num_classes and drouput defined in the previous
# cell
model = getattr(models, model_name)(num_classes=num_classes)
s_epoch=load_model(model_name+suffix,model)
model.post=models.Topk(k=7)
model.Dconvs0[5]=torch.nn.Identity()
#print(model)
#model= shufflenet_v2_x1_0(pretrained=True)
print(f"model has :{sum(p.numel() for p in model.parameters())/1e6} M parameters ")
#model.half()

def loss(output,target):
    #target=F.one_hot(target, num_classes).float()
    return F.cross_entropy(output, target,label_smoothing=0.01)

model.cuda()
lo, acc= valid_one_epoch(data_loaders['valid'], model, loss)

print("Name:{}\tSuffix: {}\tLoss: {:.6f}\tAcc: {:.3f}\t5Acc: {:.3f} ".format(model_name,suffix,lo,acc[0], acc[1]))
