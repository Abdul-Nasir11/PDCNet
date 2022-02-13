import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F



__all__ = ["PDCNet"]

class Downsample(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=32,
                 kernel_size=3,
                 stride = 2,
                 bias=False,
                 relu=True,
                 out_channels_div= [11,9,9] 
            ):
        super(Downsample,self).__init__()


        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        self.conv2d_d1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_div[0],kernel_size=kernel_size, stride=stride, padding=1, bias=bias, dilation=1)
        self.conv2d_d2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_div[1],kernel_size=kernel_size, stride=stride, padding=2, bias=bias, dilation=2)
        self.conv2d_d5 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_div[2],kernel_size=kernel_size, stride=stride, padding=5, bias=bias, dilation=5)
        self.ext_branch = nn.MaxPool2d(kernel_size, stride=2, padding=1)
        self.activation = activation
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        # print('input:', input.shape)
        main1_d1= self.conv2d_d1(input)
        # print('dialation 1:',main1_d1.shape)
        main2_d2 = self.conv2d_d2(input)
        # print('dialation 2', main2_d2.shape)
        main3_d5 = self.conv2d_d5(input)
        # print('dialation 5:',main3_d5.shape)
        # ext1=self.half(input)
        ext1 = self.ext_branch(input)
        # print(ext1.shape)
        
        
        
        out = torch.cat((main1_d1,main2_d2,main3_d5,ext1), dim=1)
        # print('out after concate', out.shape)
        out = self.batch_norm(out)
        # out = self.activation(out)
        out = nn.functional.relu(out)

        # print('final', out)


        return out



class Regular(nn.Module):
    def __init__(self,
                 in_channels=32,
                 out_channels=32,
                #  internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dropout_prob=0.,
                 stride = 1,
                 bias=False,
                 relu=True, 
                 out_channels_div = [11,11,10]
                 
                 ):
        super(Regular,self).__init__()


        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        self.conv2d_d1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_div[0],kernel_size=kernel_size, stride=stride, padding=1, bias=bias, dilation=1)
        self.conv2d_d2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_div[1],kernel_size=kernel_size, stride=stride, padding=2, bias=bias, dilation=2)
        self.conv2d_d5 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_div[2],kernel_size=kernel_size, stride=stride, padding=5, bias=bias, dilation=5)
        # self.half = nn.MaxPool2d(2, stride=2)
        self.activation = activation
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        # print('input:', input.shape)
        main1_d1= self.conv2d_d1(input)
        # print('dialation 1:',main1_d1.shape)
        main2_d2 = self.conv2d_d2(input)
        # print('dialation 2', main2_d2.shape)
        main3_d5 = self.conv2d_d5(input)
        # print('dialation 5:',main3_d5.shape)
        # ext1=self.half(input)
        
        
        
        out = torch.cat((main1_d1,main2_d2,main3_d5), dim=1)
        # print('out after concate', out.shape)
        out = self.batch_norm(out)
        # out = self.activation(out)
        out = nn.functional.relu(out)

        # print('final', out)


        return out


class Encoder(nn.Module):
    def __init__(self, encoder_relu=False, decoder_relu=True):
        super().__init__()

        
        self.downsample_0 = Downsample(in_channels=3,out_channels=16, kernel_size=3, relu=True, out_channels_div=[5,4,4])
        self.regular_0 = Regular(in_channels=16, out_channels=16, kernel_size=3, out_channels_div=[6,5,5])
        self.regular_0x2 = Regular(in_channels=16, out_channels=16, kernel_size=3, out_channels_div=[6,5,5])

        self.downsample_1 = Downsample(in_channels=16,out_channels=32, kernel_size=3, relu=True, out_channels_div=[6,5,5])
        self.regular_1 = Regular(in_channels=32, out_channels=32, kernel_size=3, out_channels_div=[11,11,10])
        self.regular_1x2 = Regular(in_channels=32, out_channels=32, kernel_size=3, out_channels_div=[11,11,10])
        
        
        
        self.downsample_2 = Downsample(in_channels=32, out_channels=64, kernel_size=3,relu=True, out_channels_div=[11,11,10])
        self.regular_2 = Regular(in_channels=64, out_channels=64,out_channels_div=[22,21,21])
        self.regular_2x2 = Regular(in_channels=64, out_channels=64,out_channels_div=[22,21,21])
        
        
        self.downsample_3 = Downsample(in_channels=64, out_channels=128, out_channels_div=[22,21,21])
        self.regular_3 = Regular(in_channels=128, out_channels=128, out_channels_div=[43,43,42])
        self.regular_4 = Regular(in_channels=128, out_channels=128, out_channels_div=[43,43,42] )

    


    def forward(self, input):
        x= self.downsample_0(input)
        x = self.regular_0(x)
        x = self.regular_0x2(x)



        x = self.downsample_1(x)
        x = self.regular_1(x)
        x = self.regular_1x2(x)
        
        x = self.downsample_2(x)
        x = self.regular_2(x)
        x = self.regular_2x2(x)
        
        x = self.downsample_3(x)
        x= self.regular_3(x)
        x= self.regular_4(x)

        return x




class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels,out_padding =0 ,kernel_size = 3, relu = False, stride=2):
        super().__init__()

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU


        
        
        self.conv = nn.ConvTranspose2d(in_channels= in_channels, 
                                        out_channels=out_channels, stride=stride,
                                        kernel_size=kernel_size, 
                                        padding=1, output_padding=1,
                                        bias=True
                                        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = activation
    def forward(self, input):
        # print('Input Shape: ', input.shape)
        output = self.conv(input)
        output = self.batch_norm(output)
        output = torch.nn.functional.relu(output)
        # print('Output Shape: ', output.shape)
        

        return output
  



class Regular_decode(nn.Module):
    def __init__(self,
                 in_channels=64,
                 out_channels=64,
                #  internal_ratio=4,
                 kernel_size=3,
                #  padding=0,
                #  dropout_prob=0.,
                 stride = 1,
                 bias=False,
                 relu=True, 
                 out_channels_div = [22,21,21]
                 
                 ):
        super().__init__()


        if relu:
            # activation = nn.ReLU()
            activation = torch.nn.functional.relu
        else:
            activation = nn.PReLU()

        self.conv2d_d1 = nn.Conv2d(in_channels=in_channels, 
                                    out_channels=out_channels_div[0],
                                    kernel_size=kernel_size, 
                                    stride=stride, 
                                    padding=1, 
                                    bias=False, 
                                    dilation=1
                                    )

        self.conv2d_d2 = nn.Conv2d(in_channels=in_channels, 
                                    out_channels=out_channels_div[1],
                                    kernel_size=kernel_size, 
                                    stride=stride, 
                                    padding=2, 
                                    bias=False, 
                                    dilation=2
                                    )

        self.conv2d_d5 = nn.Conv2d(in_channels=in_channels, 
                                    out_channels=out_channels_div[2],
                                    kernel_size=kernel_size, 
                                    stride=stride, padding=5, 
                                    bias=False, 
                                    dilation=5
                                    )

        # self.half = nn.MaxPool2d(2, stride=2)

        self.activation = activation
        
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        # print('input:', input.shape)
        main1_d1= self.conv2d_d1(input)
        # print('dialation 1:',main1_d1.shape)
        main2_d2 = self.conv2d_d2(input)
        #  print('dialation 2', main2_d2.shape)
        main3_d5 = self.conv2d_d5(input)
        #  print('dialation 5:',main3_d5.shape)
        
        
        
        out = torch.cat((main1_d1,main2_d2,main3_d5), dim=1)
        # print('out after concate', out.shape)
        out = self.batch_norm(out)
        out = torch.nn.functional.relu(out)
        # out = self.activation(out)
        # print('final', out)


        return out















class Decoder(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.upsample_1 = Upsample(in_channels=128, out_channels=64)
        self.regular_de_1 = Regular_decode(in_channels=64, out_channels_div=[22,21,21])
        
        self.upsample_2 = Upsample(in_channels=64, out_channels=32)
        self.regular_de_2 = Regular_decode(in_channels=32, out_channels=32, out_channels_div=[11,11,10])

        
        self.upsample_3 = Upsample(in_channels=32, out_channels=16)



        self.regular_de_3 = Regular_decode(in_channels=16, out_channels=16, out_channels_div=[6,5,5])
        self.upsample_4 = Upsample(in_channels=16, out_channels=num_classes)
        


    def forward(self, input):
        # print('THis is the output:', input.shape)
        x = self.upsample_1(input)
        x = self.regular_de_1(x)
        x = self.upsample_2(x)
        x = self.regular_de_2(x)
        x = self.upsample_3(x)
        x = self.regular_de_3(x)
        x = self.upsample_4(x)
        
        

        return x




class PDCNet(nn.Module):
    def __init__(self, num_classes =3):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder(num_classes= num_classes)

    def forward(self, input):

        check_input = input[0][0].shape
        check_input =check_input[0]
        x = self.encoder(input)
        x= self.decoder(x)


        return x



if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PDCNet().to(device)
    # summary(model,(3,480,640))
    # summary(model,(3,368,640))
    summary(model,(3,720,1280))
    # summary(model,(3,360,480))
    # print(model)


