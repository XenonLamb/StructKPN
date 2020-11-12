import caffe
import numpy as np
from red_cnn import RedCNN
import torch
import torch.nn as nn

def redcnn_caffe2pytorch(caffenet):
    pytorchnet = RedCNN()
    pytorchnet.eval()
    with torch.no_grad():
        pytorchnet.conv1.weight = nn.Parameter(torch.tensor(caffenet.params['conv1'][0].data))
        pytorchnet.conv1.bias   = nn.Parameter(torch.tensor(caffenet.params['conv1'][1].data))
        pytorchnet.conv2.weight = nn.Parameter(torch.tensor(caffenet.params['conv2'][0].data))
        pytorchnet.conv2.bias   = nn.Parameter(torch.tensor(caffenet.params['conv2'][1].data))
        pytorchnet.conv3.weight = nn.Parameter(torch.tensor(caffenet.params['conv3'][0].data))
        pytorchnet.conv3.bias   = nn.Parameter(torch.tensor(caffenet.params['conv3'][1].data))
        pytorchnet.conv4.weight = nn.Parameter(torch.tensor(caffenet.params['conv4'][0].data))
        pytorchnet.conv4.bias   = nn.Parameter(torch.tensor(caffenet.params['conv4'][1].data))
        pytorchnet.conv5.weight = nn.Parameter(torch.tensor(caffenet.params['conv5'][0].data))
        pytorchnet.conv5.bias   = nn.Parameter(torch.tensor(caffenet.params['conv5'][1].data))

        # pytorchnet.deconv5.weight = nn.Parameter(torch.tensor(caffenet.params['deconv5'][0].data))
        # pytorchnet.deconv5.bias   = nn.Parameter(torch.tensor(caffenet.params['deconv5'][1].data))
        pytorchnet.deconv5.weight = nn.Parameter(torch.tensor(caffenet.params['deconv5'][0].data))
        pytorchnet.deconv5.bias   = nn.Parameter(torch.tensor(caffenet.params['deconv5'][1].data))
        


        pytorchnet.deconv4.weight = nn.Parameter(torch.tensor(caffenet.params['deconv4'][0].data))
        pytorchnet.deconv4.bias   = nn.Parameter(torch.tensor(caffenet.params['deconv4'][1].data))
        pytorchnet.deconv3.weight = nn.Parameter(torch.tensor(caffenet.params['deconv3'][0].data))
        pytorchnet.deconv3.bias   = nn.Parameter(torch.tensor(caffenet.params['deconv3'][1].data))
        pytorchnet.deconv2.weight = nn.Parameter(torch.tensor(caffenet.params['deconv2'][0].data))
        pytorchnet.deconv2.bias   = nn.Parameter(torch.tensor(caffenet.params['deconv2'][1].data))
        pytorchnet.deconv1.weight = nn.Parameter(torch.tensor(caffenet.params['deconv1'][0].data))
        pytorchnet.deconv1.bias   = nn.Parameter(torch.tensor(caffenet.params['deconv1'][1].data))
    
    return pytorchnet

if __name__ == '__main__':
    caffe.set_mode_gpu()
    import pydicom
    ds = pydicom.read_file('third_party_red_cnn/L506_QD_3_1.CT.0003.0035.2015.12.22.20.45.42.541197.358791561.IMA')
    img = ds.pixel_array.astype(np.float32) / 3000.0
    img = img.astype(np.float32)
     
    prototxt = 'third_party_red_cnn/Red_CNN.prototxt'
    caffemodel = 'third_party_red_cnn/Red_CNN.caffemodel'
    caffenet = caffe.Net(prototxt, caffemodel, caffe.TEST)
    # print(caffenet.params)

    caffenet.blobs['data'].data[0,0,:,:] = img
    # net.blobs['data'].data[:] = matcaffe_in

    out_dict = caffenet.forward()
    out = out_dict['eltwise2']

    img_tensor = torch.tensor(img, requires_grad=True)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).cuda()
    
    pytorchnet = redcnn_caffe2pytorch(caffenet)
    torch.save({
        'state_dict': pytorchnet.state_dict(),
        'name': 'red_cnn',
        'info': 'converted from caffemodel, might have up to 1e-7 error'
    },'third_party_red_cnn/redcnn_caffeconverted_ckpt.t7')


    # #### to check if the conversion is successful
    # pytorchnet = pytorchnet.cuda()
    # pytorchnet.eval()
    # with torch.no_grad():
    #     # outpytorch = pytorchnet.forward(img_tensor)
    #     # outpytorch = outpytorch_tuple[0]
    #     # outpytorch = outpytorch.detach().cpu().numpy()

    #     outpytorch_tuple = pytorchnet.forward(img_tensor)
    #     comparing_list = ['conv5r', 'deconv5', 'eltwise', 'eltwiser', 
    #         'deconv4', 'deconv3', 'deconv2', 'deconv1', 'eltwise2']
    #     comparing_dict = dict([])
    #     i = 0
    #     for key in comparing_list:
    #         temp = outpytorch_tuple[i]
    #         temp = temp.detach().cpu().numpy()
    #         comparing_dict[key] = temp
    #         i += 1

    # for key in comparing_list:
    #     print('maximum different of blob {}: {}'.format(key, (caffenet.blobs[key].data - comparing_dict[key]).max()))


    # from matplotlib import pyplot as plt
    # plt.imsave('RED-CNN_L506_35_pycaffe.png', out[0,0,:,:], cmap='gray', vmin=835.0/3000.0, vmax=1275.0/3000.0)
    # # plt.imshow(out[0,0,:,:], cmap='gray', vmin=835.0/3000.0, vmax=1275.0/3000.0)
    # # plt.show()
    # # plt.imshow(outpytorch[0,0,:,:], cmap='gray', vmin=835.0/3000.0, vmax=1275.0/3000.0)
    # # plt.show()
    # plt.imsave('RED-CNN_L506_35_pytorch.png', outpytorch_tuple[-1][0,0,:,:], cmap='gray', vmin=835.0/3000.0, vmax=1275.0/3000.0)
    



    





