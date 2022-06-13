import torch
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Bilinear
import torchsummary
INPUT_IMAGE_SHAPE = (512, 224)


class BottleNeck(torch.nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()  

        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=1, bias=False, dilation=1)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)

        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False, dilation=1)
        self.bn2 = torch.nn.BatchNorm2d(planes)

        # self.conv3 = torch.nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False, dilation=1)
        self.conv3 = torch.nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False, dilation=1)
        self.bn3 = torch.nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = out.clone() + residual
        out = self.relu(out)

        return out




class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()  
        self.layers = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False))
        self.convs.append(torch.nn.Conv2d(64, 128, kernel_size=(3,3), padding=1))
        self.convs.append(torch.nn.Conv2d(128, 256, kernel_size=(3,3), padding=1))
   
        self.bns = torch.nn.ModuleList()
        
        self.bns.append(torch.nn.BatchNorm2d(64))
        self.bns.append(torch.nn.BatchNorm2d(128))
        self.bns.append(torch.nn.BatchNorm2d(256))


        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)

        self.layers.append(BottleNeck(64, 64))
        self.layers.append(BottleNeck(128, 128))
        self.layers.append(BottleNeck(256, 256))


    def forward(self, x):

        for i in range(len(self.convs)):
            x = self.convs[i](x)
            x = self.bns[i](x)
            x = self.layers[i](x)
            x = self.relu(x)
            x = self.maxpool(x)

        return x

class myNN(torch.nn.Module):
    def __init__(self):
        super().__init__()  
        self.backbone = Backbone()

        self.reduce_features = torch.nn.ModuleList()
    
        self.reduce_features.append(torch.nn.Conv2d(256, 128, kernel_size=1))
        self.reduce_features.append(torch.nn.Conv2d(128, 64, kernel_size=1))
        self.reduce_features.append(torch.nn.Conv2d(64, 1, kernel_size=1))

        self.conv = torch.nn.Sequential(
            #3 224 128
            torch.nn.Conv2d(3, 64, 3, padding=1),torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64, 64, 3, padding=1),torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(64, 128, 3, padding=1),torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(128, 128, 3, padding=1),torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool2d(2, 2),
            #128 56 32
            torch.nn.Conv2d(128, 256, 3, padding=1),torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(256, 256, 3, padding=1),torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(256, 256, 3, padding=1),torch.nn.LeakyReLU(0.2),
        )

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def forward(self, x):
        #backbone_out = self.backbone(x)

        semiOut = self.conv(x)
        for reduce_layer in self.reduce_features:
            semiOut = reduce_layer(semiOut)
            semiOut = F.interpolate(semiOut, (semiOut.shape[2] * 2, semiOut.shape[3] * 2), mode='bilinear', align_corners=False)

        mask_out = semiOut.contiguous()
        
        mask_out = F.interpolate(mask_out, (INPUT_IMAGE_SHAPE[1], INPUT_IMAGE_SHAPE[0]), mode='bilinear', align_corners=False).squeeze(0)


        return mask_out

if __name__=="__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(777)

    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    model = myNN()
    model = torch.nn.DataParallel(model)
    model.to(device)

    print(model)
    torchsummary.summary(model, input_size=(3, 224, 512))

