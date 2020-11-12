import torch
import torch.nn as nn

class RedCNN(nn.Module):
    def __init__(self):
        super(RedCNN, self).__init__()
        self.relu  = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(1,96,5)
        self.conv2 = nn.Conv2d(96,96,5)
        self.conv3 = nn.Conv2d(96,96,5)
        self.conv4 = nn.Conv2d(96,96,5)
        self.conv5 = nn.Conv2d(96,96,5)

        self.deconv5 = nn.ConvTranspose2d(96,96,5)
        self.deconv4 = nn.ConvTranspose2d(96,96,5)
        self.deconv3 = nn.ConvTranspose2d(96,96,5)
        self.deconv2 = nn.ConvTranspose2d(96,96,5)
        self.deconv1 = nn.ConvTranspose2d(96,1,5)

    def forward(self, x):
        en1 = self.conv1(x)
        en1 = self.relu(en1)
        en2 = self.conv2(en1)
        en2 = self.relu(en2)
        en3 = self.conv3(en2)
        en3 = self.relu(en3)
        en4 = self.conv4(en3)
        en4 = self.relu(en4)
        en5 = self.conv5(en4)
        en5 = self.relu(en5)

        # de4 = self.deconv5(en5)
        # de4 += en4
        # de4 = self.relu(de4)
        # de3 = self.deconv4(de4)
        # de3 = self.relu(de3)
        # de2 = self.deconv3(en3)
        # de2 += en2
        # de2 = self.relu(de2)
        # de1 = self.deconv2(de2)
        # de1 = self.relu(de1)
        # out = self.deconv1(de1)
        # out += x
        # out = self.relu(out)
        # return out

        #### debug
        conv5r = en5
        deconv5 = self.deconv5(conv5r)
        eltwise = deconv5 + en4
        eltwiser = self.relu(eltwise)
        deconv4 = self.deconv4(eltwiser)
        deconv4r = self.relu(deconv4)
        deconv3 = self.deconv3(deconv4r)
        eltwise1 = en2 + deconv3
        eltwise1r = self.relu(eltwise1)
        deconv2 = self.deconv2(eltwise1r)
        deconv2r = self.relu(deconv2)
        deconv1 = self.deconv1(deconv2r)
        eltwise2 = x + deconv1
        eltwise2 = self.relu(eltwise2)
        return eltwise2
        # return conv5r, deconv5, eltwise, eltwiser, deconv4, deconv3, deconv2, deconv1, eltwise2
        #### debug


class RedCNN(nn.Module):
    def __init__(self):
        super(RedCNN, self).__init__()
        self.relu  = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(1,96,5)
        self.conv2 = nn.Conv2d(96,96,5)
        self.conv3 = nn.Conv2d(96,96,5)
        self.conv4 = nn.Conv2d(96,96,5)
        self.conv5 = nn.Conv2d(96,96,5)

        self.deconv5 = nn.ConvTranspose2d(96,96,5)
        self.deconv4 = nn.ConvTranspose2d(96,96,5)
        self.deconv3 = nn.ConvTranspose2d(96,96,5)
        self.deconv2 = nn.ConvTranspose2d(96,96,5)
        self.deconv1 = nn.ConvTranspose2d(96,1,5)

    def forward(self, x, temperature=1.0):
        en1 = self.conv1(x)
        en1 = self.relu(en1)
        en2 = self.conv2(en1)
        en2 = self.relu(en2)
        en3 = self.conv3(en2)
        en3 = self.relu(en3)
        en4 = self.conv4(en3)
        en4 = self.relu(en4)
        en5 = self.conv5(en4)
        en5 = self.relu(en5)

        # de4 = self.deconv5(en5)
        # de4 += en4
        # de4 = self.relu(de4)
        # de3 = self.deconv4(de4)
        # de3 = self.relu(de3)
        # de2 = self.deconv3(en3)
        # de2 += en2
        # de2 = self.relu(de2)
        # de1 = self.deconv2(de2)
        # de1 = self.relu(de1)
        # out = self.deconv1(de1)
        # out += x
        # out = self.relu(out)
        # return out

        #### debug
        conv5r = en5
        deconv5 = self.deconv5(conv5r)
        eltwise = deconv5 + en4
        eltwiser = self.relu(eltwise)
        deconv4 = self.deconv4(eltwiser)
        deconv4r = self.relu(deconv4)
        deconv3 = self.deconv3(deconv4r)
        eltwise1 = en2 + deconv3
        eltwise1r = self.relu(eltwise1)
        deconv2 = self.deconv2(eltwise1r)
        deconv2r = self.relu(deconv2)
        deconv1 = self.deconv1(deconv2r)
        eltwise2 = x + deconv1
        eltwise2 = self.relu(eltwise2)
        return eltwise2
        # return conv5r, deconv5, eltwise, eltwiser, deconv4, deconv3, deconv2, deconv1, eltwise2
        #### debug

class RedCNNWIDE(nn.Module):
    def __init__(self):
        super(RedCNNWIDE, self).__init__()
        self.relu  = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(1,146,5)
        self.conv2 = nn.Conv2d(146,146,5)
        self.conv3 = nn.Conv2d(146,146,5)
        self.conv4 = nn.Conv2d(146,146,5)
        self.conv5 = nn.Conv2d(146,146,5)

        self.deconv5 = nn.ConvTranspose2d(146,146,5)
        self.deconv4 = nn.ConvTranspose2d(146,146,5)
        self.deconv3 = nn.ConvTranspose2d(146,146,5)
        self.deconv2 = nn.ConvTranspose2d(146,146,5)
        self.deconv1 = nn.ConvTranspose2d(146,1,5)

    def forward(self, x,temperature=1.0):
        en1 = self.conv1(x)
        en1 = self.relu(en1)
        en2 = self.conv2(en1)
        en2 = self.relu(en2)
        en3 = self.conv3(en2)
        en3 = self.relu(en3)
        en4 = self.conv4(en3)
        en4 = self.relu(en4)
        en5 = self.conv5(en4)
        en5 = self.relu(en5)

        # de4 = self.deconv5(en5)
        # de4 += en4
        # de4 = self.relu(de4)
        # de3 = self.deconv4(de4)
        # de3 = self.relu(de3)
        # de2 = self.deconv3(en3)
        # de2 += en2
        # de2 = self.relu(de2)
        # de1 = self.deconv2(de2)
        # de1 = self.relu(de1)
        # out = self.deconv1(de1)
        # out += x
        # out = self.relu(out)
        # return out

        #### debug
        conv5r = en5
        deconv5 = self.deconv5(conv5r)
        eltwise = deconv5 + en4
        eltwiser = self.relu(eltwise)
        deconv4 = self.deconv4(eltwiser)
        deconv4r = self.relu(deconv4)
        deconv3 = self.deconv3(deconv4r)
        eltwise1 = en2 + deconv3
        eltwise1r = self.relu(eltwise1)
        deconv2 = self.deconv2(eltwise1r)
        deconv2r = self.relu(deconv2)
        deconv1 = self.deconv1(deconv2r)
        eltwise2 = x + deconv1
        eltwise2 = self.relu(eltwise2)
        return eltwise2
        # return conv5r, deconv5, eltwise, eltwiser, deconv4, deconv3, deconv2, deconv1, eltwise2
        #### debug


class RedCNNLARGE(nn.Module):
    def __init__(self):
        super(RedCNNLARGE, self).__init__()
        self.relu  = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(1,96,5)
        self.conv2 = nn.Conv2d(96,96,5)
        self.conv3 = nn.Conv2d(96,96,5)
        self.conv4 = nn.Conv2d(96,96,5)
        self.conv5 = nn.Conv2d(96,96,5)
        self.conv6 = nn.Conv2d(96, 96, 5)
        self.conv7 = nn.Conv2d(96, 96, 5)
        self.conv8 = nn.Conv2d(96, 96, 5)
        self.conv9 = nn.Conv2d(96, 96, 5)
        self.conv10 = nn.Conv2d(96, 96, 5)
        self.conv11 = nn.Conv2d(96, 96, 5)

        self.deconv11 = nn.ConvTranspose2d(96, 96, 5)
        self.deconv10 = nn.ConvTranspose2d(96, 96, 5)
        self.deconv9 = nn.ConvTranspose2d(96, 96, 5)
        self.deconv8 = nn.ConvTranspose2d(96, 96, 5)
        self.deconv7 = nn.ConvTranspose2d(96, 96, 5)
        self.deconv6 = nn.ConvTranspose2d(96, 96, 5)
        self.deconv5 = nn.ConvTranspose2d(96,96,5)
        self.deconv4 = nn.ConvTranspose2d(96,96,5)
        self.deconv3 = nn.ConvTranspose2d(96,96,5)
        self.deconv2 = nn.ConvTranspose2d(96,96,5)
        self.deconv1 = nn.ConvTranspose2d(96,1,5)

    def forward(self, x, temperature=1.0):
        en1 = self.conv1(x)
        en1 = self.relu(en1)
        en2 = self.conv2(en1)
        en2 = self.relu(en2)
        en3 = self.conv3(en2)
        en3 = self.relu(en3)
        en4 = self.conv4(en3)
        en4 = self.relu(en4)
        en5 = self.conv5(en4)
        en5 = self.relu(en5)
        en6 = self.conv6(en5)
        en6 = self.relu(en6)
        en7 = self.conv7(en6)
        en7 = self.relu(en7)
        en8 = self.conv8(en7)
        en8 = self.relu(en8)
        en9 = self.conv9(en8)
        en9 = self.relu(en9)
        en10 = self.conv10(en9)
        en10 = self.relu(en10)
        en11 = self.conv11(en10)
        en11 = self.relu(en11)

        # de4 = self.deconv5(en5)
        # de4 += en4
        # de4 = self.relu(de4)
        # de3 = self.deconv4(de4)
        # de3 = self.relu(de3)
        # de2 = self.deconv3(en3)
        # de2 += en2
        # de2 = self.relu(de2)
        # de1 = self.deconv2(de2)
        # de1 = self.relu(de1)
        # out = self.deconv1(de1)
        # out += x
        # out = self.relu(out)
        # return out

        #### debug
        conv11r = en11
        deconv11 = self.deconv11(conv11r)
        eltwise = deconv11 + en10
        eltwiser = self.relu(eltwise)
        deconv10 = self.deconv10(eltwiser)
        deconv10r = self.relu(deconv10)
        deconv9 = self.deconv9(deconv10r)
        eltwise = deconv9 + en8
        eltwiser = self.relu(eltwise)

        deconv8 = self.deconv8(eltwiser)
        deconv8r = self.relu(deconv8)
        deconv7 = self.deconv7(deconv8r)
        eltwise = deconv7 + en6
        eltwiser = self.relu(eltwise)

        deconv6 = self.deconv6(eltwiser)
        deconv6r = self.relu(deconv6)
        deconv5 = self.deconv5(deconv6r)
        eltwise = deconv5 + en4


        #conv5r = en5
        #deconv5 = self.deconv5(conv5r)
        #eltwise = deconv5 + en4
        eltwiser = self.relu(eltwise)
        deconv4 = self.deconv4(eltwiser)
        deconv4r = self.relu(deconv4)
        deconv3 = self.deconv3(deconv4r)
        eltwise1 = en2 + deconv3
        eltwise1r = self.relu(eltwise1)
        deconv2 = self.deconv2(eltwise1r)
        deconv2r = self.relu(deconv2)
        deconv1 = self.deconv1(deconv2r)
        eltwise2 = x + deconv1
        eltwise2 = self.relu(eltwise2)
        return eltwise2
        # return conv5r, deconv5, eltwise, eltwiser, deconv4, deconv3, deconv2, deconv1, eltwise2
        #### debug

def InitializeWith3mmWeight(redcnn, netOpt):
    ckpt = torch.load(netOpt['initCkptDir'])
    redcnn.load_state_dict(ckpt['state_dict'])
    return redcnn



if __name__=='__main__':
    #### debug network implementation
    # x = torch.zeros(5,1,512,512)
    # x.requires_grad=True
    # x = x.cuda()
    # net = RedCNN()
    # net.eval()
    # net = net.cuda()
    # net.forward(x)
    #### debug network implementation

    redcnn = RedCNN()
    ckpt = torch.load('third_party_red_cnn/redcnn_caffeconverted_ckpt.t7')
    redcnn.load_state_dict(ckpt['state_dict'])

    #### check if this pytorch redcnn is correct
    redcnn = redcnn.cuda()
    redcnn.eval()

    import pydicom
    import numpy as np
    ds = pydicom.read_file('third_party_red_cnn/L506_QD_3_1.CT.0003.0035.2015.12.22.20.45.42.541197.358791561.IMA')
    img = ds.pixel_array.astype(np.float32) / 3000.0
    img = img.astype(np.float32)
    img_tensor = torch.tensor(img, requires_grad=True)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).cuda()

    with torch.no_grad():
        out = redcnn.forward(img_tensor)

    out = out.detach().cpu().numpy()

    from matplotlib import pyplot as plt
    plt.imshow(out[0,0,:,:], cmap='gray', vmin=835.0/3000.0, vmax=1275.0/3000.0)
    plt.show()