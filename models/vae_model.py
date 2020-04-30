"""Model class template
This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
from .base_model import BaseModel
from . import networks
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable

class VAE(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size, device):
        super(VAE, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        self.is_cuda = True
        if device == torch.device('cpu'):
            self.is_cuda = False
        
        # encoder
        self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.e4 = nn.Conv2d(ndf*4, ndf*4, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf*4)

        # changed 12 = 4
        self.e5 = nn.Conv2d(ndf*4, ndf*4, 12, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf*4)

        self.e6 = nn.Conv2d( ndf*8, ndf*8, 4, 2, 1)
        self.bn6 = nn.BatchNorm2d(ndf*8)

        #changed  8 = 4
        self.fc1 = nn.Linear(ndf*4*4*4, latent_variable_size)
        self.fc2 = nn.Linear(ndf*4*4*4, latent_variable_size)

        # decoder
        #changed 4 = 8
        self.d1 = nn.Linear(latent_variable_size, ngf*4*2*4*4)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        #changed 4 = 8
        self.d2 = nn.Conv2d(ngf*4*2, ngf*4, 3, 1)
        self.bn6 = nn.BatchNorm2d(ngf*4, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        #changed 4 = 8, 2 = 4
        self.d3 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
        self.bn7 = nn.BatchNorm2d(ngf*2, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        # changed 4 = 2, none = 2
        self.d4 = nn.Conv2d(ngf*2, ngf, 3, 1)
        self.bn8 = nn.BatchNorm2d(ngf, 1.e-3)
        # 5, 256, 32, 32

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        #changed ngf = ngf * 2
        self.d5 = nn.Conv2d(ngf, ngf, 3, 1)
        self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)
        # 5, 256, 64, 64

        self.up6 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd6 = nn.ReplicationPad2d(1)
        self.d7 = nn.Conv2d(ngf, ngf, 3, 1)
        self.bn10 = nn.BatchNorm2d(ngf, 1.e-3)
        # 5, 256, 128, 128

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(ngf, nc, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        #h6 = self.leakyrelu(self.bn6(self.e6(h5)))
        #changed 4 = 8
        h6 = h5.view(-1, self.ndf*4*4*4)

        return self.fc1(h6), self.fc2(h6)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        #changed 4 = 8
        h1 = h1.view(-1, self.ngf*4*2, 4, 4)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))
        h6 = self.leakyrelu(self.bn10(self.d7(self.pd6(self.up6(h5)))))
        #changed h6 = h5
        # return self.sigmoid( self.d6(self.pd5(self.up5(h6))) )
        return self.tanh(self.d6(self.pd5(self.up5(h6))))

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar

class VaeModel(BaseModel):
    """ 
    This class implements a custom conditional VAE model, for learning a mapping 
    from input images to output images given paired data. 
    We build off of ideas in the pix2pix paper and how these are different from conditional GANs
    which are conditioned on labels. Here, we try to replicate a conditional VAE equivalent
    where we condition on the input image in the decoder (instead of a label) and try to use 
    an appropriate loss function to generate the target image. 
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.
        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser.set_defaults(dataset_mode='aligned')  # You can rewrite default values for this model. For example, this model usually uses aligned dataset as its dataset.
        # if is_train:
        #     parser.add_argument('--lambda_regression', type=float, default=1.0, help='weight for the regression loss')  # You can define new arguments for this model.

        return parser

    def __init__(self, opt):
        """Initialize this model class.
        Parameters:
            opt -- training/test options
        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        # self.loss_names = ['G', 'G_GAN', 'reconstruction', 'KLD', 'D_real', 'D_fake']
        self.loss_names = ['G', 'G_GAN', 'reconstruction', 'D_real', 'D_fake']
        
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['data_A', 'fake_B', 'data_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        
        self.netVAE = VAE(3, 256, 256, 500, self.device)
        if len(self.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.netVAE.to(self.gpu_ids[0])
            self.netVAE = torch.nn.DataParallel(self.netVAE, self.gpu_ids)  # multi-GPUs

        # initialize
        # init.normal_(self.netVAE.weight.data, 0.0, 0.02)
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.normal_(m.weight.data, 0.0, 0.02)                
            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

        self.netVAE.apply(init_func)

        if self.isTrain:
            self.model_names = ['VAE', 'D']
        else:  # during test time, only load G
            self.model_names = ['VAE']

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # only defined during training time
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            # We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionGAN = networks.GANLoss('lsgan').to(self.device) 
            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            self.optimizer_G = torch.optim.Adam(self.netVAE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        AtoB = self.opt.direction == 'AtoB'  # use <direction> to swap data_A and data_B
        self.data_A = input['A' if AtoB else 'B'].to(self.device)  # get image data A
        self.data_B = input['B' if AtoB else 'A'].to(self.device)  # get image data B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']  # get image paths

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        # self.output = self.netG(self.data_A)  # generate output image given the input data_A

        self.fake_B, self.mu, self.variance = self.netVAE.forward(self.data_A) 
    
    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_reconstruction = self.criterionL1(self.fake_B, self.data_B) * 100 #CC hardcoded lambda
        # KLD_loss_element = self.mu.pow(2).add_(self.variance.exp()).mul_(-1).add_(1).add_(self.variance)
        # self.loss_KLD = torch.sum(KLD_loss_element).mul_(-0.5)
        
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.data_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # KLD_loss = -0.5 * torch.sum(1 + self.variance - self.mu.pow(2) - self.variance.exp())
        self.loss_G = self.loss_G_GAN + self.loss_reconstruction # + self.loss_KLD
        self.loss_G.backward()       # calculate gradients of network G w.r.t. loss_G
    
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.data_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.data_A, self.data_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)

        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights

        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights