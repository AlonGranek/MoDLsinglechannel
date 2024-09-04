import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.complex_utils as cplx
from utils.transforms import SenseModel,SenseModel_single
from utils.flare_utils import ConjGrad


############### Transformer code ################

class PatchEmbedding(nn.Module):
    def __init__(self, img_height, img_width, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.num_patches = (img_height // patch_size) * (img_width // patch_size)
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_patches):
        super(PositionalEncoding, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
    
    def forward(self, x):
        return x + self.pos_embed

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, forward_expansion):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * forward_expansion)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        return self.encoder(x)

class ConvDecoder(nn.Module):
    def __init__(self, embed_dim, patch_size, img_height, img_width, out_channels):
        super(ConvDecoder, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.img_height = img_height
        self.img_width = img_width
        self.num_patches = (img_height // patch_size) * (img_width // patch_size)
        self.avg_pool = nn.AvgPool2d(kernel_size=patch_size, stride=16)

        # Ensure stride and padding are set correctly to match the dataset size
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=patch_size, stride=patch_size, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 2, out_channels, kernel_size=patch_size, stride=patch_size, padding=0)
        )

    def forward(self, x):
        B, num_patches, embed_dim = x.size()
        x = x.transpose(1, 2)  # (B, embed_dim, num_patches)
        x = x.view(B, embed_dim, self.img_height // self.patch_size, self.img_width // self.patch_size)  # (B, embed_dim, H/patch_size, W/patch_size)
        return self.avg_pool(self.deconv(x))



class TransformerImage2Image(nn.Module):
    def __init__(self, img_height=256, img_width=160, patch_size=16, in_channels=4, out_channels=2, embed_dim=768, num_heads=8, num_layers=6, forward_expansion=4):
        super(TransformerImage2Image, self).__init__()
        self.patch_embed = PatchEmbedding(img_height, img_width, patch_size, in_channels, embed_dim)
        self.pos_embed = PositionalEncoding(embed_dim, (img_height // patch_size) * (img_width // patch_size))
        self.transformer_encoder = TransformerEncoder(embed_dim, num_heads, num_layers, forward_expansion)
        self.decoder = ConvDecoder(embed_dim, patch_size, img_height, img_width, out_channels)
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x)
        return x


################ CNN spattial ##################
class Operator(torch.nn.Module):
    def __init__(self, A):
        super(Operator, self).__init__()
        self.operator = A

    def forward(self, x):
        return self.operator(x)

    def adjoint(self, x):
        return self.operator(x, adjoint=True)

    def normal(self, x):
        out = self.adjoint(self.forward(x))
        return out
    

class LeakyReLUConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(LeakyReLUConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.lrelu = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        return self.lrelu(self.conv(x))

class mRCAB(nn.Module):
    def __init__(self, in_channels):
        super(mRCAB, self).__init__()
        self.conv1 = LeakyReLUConv(in_channels, in_channels)
        self.conv2 = LeakyReLUConv(in_channels, in_channels)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        ca = self.ca(res)
        return x + res * ca

class ResidualGroup(nn.Module):
    def __init__(self, in_channels, num_blocks):
        super(ResidualGroup, self).__init__()
        self.blocks = nn.Sequential(*[mRCAB(in_channels) for _ in range(num_blocks)])
        
    def forward(self, x):
        return x + self.blocks(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SubPixelConv(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=1):
        super(SubPixelConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))

class MyNetwork(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, num_residual_groups=3, num_blocks=5):
        super(MyNetwork, self).__init__()
        self.initial_conv = LeakyReLUConv(in_channels, 64)
        self.residual_groups1 = ResidualGroup(64, num_blocks)
        self.residual_groups2 = ResidualGroup(64, num_blocks)
        self.residual_groups3 = ResidualGroup(64, num_blocks)
        self.mrcabs = mRCAB(64) #128
        self.spatial_attention = SpatialAttention()
        self.sub_pixel_conv = SubPixelConv(64, 64) #(128,64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.sub_pixel_conv_end = SubPixelConv(64, 64,upscale_factor=2)
        
    def forward(self, kspace, reference_image,init_image=None, mask=None,iter=0):
        #if mask is None:
        #    mask = cplx.get_mask(kspace)
        #kspace *= mask

        # Get data dimensions
        dims = tuple(kspace.size())
        """
        # Declare signal model
        A = SenseModel_single(weights=mask)
        Sense = Operator(A)
        # Compute zero-filled image reconstruction
        x = Sense.adjoint(kspace)
        x = x.permute(0, 3, 1, 2)
        """
        x = kspace
        #print(f'After adjoint: {x.shape}')
        #trilnear = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)


        ## input part1
        x = self.initial_conv(x)
        #print(f'After initial_conv: {x.shape}')
        res1 = self.residual_groups1(x)
        #print(f'After res1: {res1.shape}')
        res2 = self.residual_groups2(res1)
        #print(f'After res2: {res2.shape}')
        res3 = self.residual_groups3(res2)
        #print(f'After res3: {res3.shape}')

        ## reference part1
        #reference_image = self.initial_conv(reference_image)
        #print(f'After initial_conv: {reference_image_ref.shape}')
        #res1_ref = self.residual_groups1(reference_image)
        #print(f'After res1: {res1_ref.shape}')
        #res2_ref = self.residual_groups2(res1_ref)
        #print(f'After res2: {res2_ref.shape}')
        #res3_ref = self.residual_groups3(res2_ref)
        #print(f'After res3: {res3_ref.shape}')

        ## Combine paths
        #combined = torch.cat([res3, res3_ref], dim=1)

        ## Combined path
        x = self.mrcabs(res3)
        x = self.spatial_attention(x) * x
        #print(f'After spatial_attention: {x.shape}')
        #if iter == 5:
        #    x = self.sub_pixel_conv_end(x)
        #    x = self.final_conv(x)
        #    trilnear = F.interpolate(kspace, scale_factor=2, mode='trilinear', align_corners=True)
        #    x = x + trilnear
        #if iter != 5:
        x = self.sub_pixel_conv(x)
            #print(f'After sub_pixel conv: {x.shape}')
        x = self.final_conv(x)

        #x = torch.cat([x, reference_image], dim=1)
        #print(f'After cat: {x.shape}')

        #x = self.final_transformer(x)
        #print(f'After transformer: {x.shape}')
        #print(f'After final conv: {x.shape}')
        #x = x.permute(0,2,3, 1)
        return x 


