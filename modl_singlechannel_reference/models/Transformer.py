import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from models.features_extract import deep_features
from models.extractor import ViTExtractor

"""
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        src = self.norm(src)
        return self.transformer_layer(src)

class TransUNet(nn.Module):
    def __init__(self, img_channels, base_channels, n_transformer_layers, d_model, nhead):
        super(TransUNet, self).__init__()
        
        self.encoder1 = ConvBlock(img_channels, base_channels)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2)
        self.encoder3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.encoder4 = ConvBlock(base_channels * 4, base_channels * 8)
        
        self.pool = nn.MaxPool2d(2)
        
        # Adjust dimensions to match the expected input size for the transformer
        self.transformer_proj = nn.Linear(base_channels * 8 * (256 // 16) * (160 // 16) * 4 , d_model*base_channels)
        self.transformer = nn.Sequential(
            *[TransformerLayer(d_model*base_channels, nhead) for _ in range(n_transformer_layers)]
        )
        
        self.up1 = UpBlock(base_channels * 8, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2)
        self.up3 = UpBlock(base_channels * 2, base_channels)

        
        self.final_conv = nn.Conv2d(base_channels, img_channels, kernel_size=1)

    def forward(self, x,reference_image):
        # Encoder
        x1 = self.encoder1(x)
        print(f'after encoder1: {x1.shape}')
        x2 = self.encoder2(self.pool(x1))
        print(f'after encoder2: {x2.shape}')
        x3 = self.encoder3(self.pool(x2))
        print(f'after encoder3: {x3.shape}')
        x4 = self.encoder4(self.pool(x3))

        # Transformer
        print(f'x4 size before reshaping: {x4.size()}')
        batch_size, channels, height, width = x4.size()
        x4 = x4.reshape(batch_size, channels * height * width)
        print(f'x4 size after reshaping: {x4.size()}')
        x4 = self.transformer_proj(x4)
        x4 = x4.unsqueeze(1)
        print(f'before transformer: {x4.size()}')
        x4 = self.transformer(x4)
        print(f'After transformer: {x4.size()}')
        x4 = x4.squeeze(1)
        
        print(f'x4 size before reshaping back: {x4.size()}')
        x4 = x4.reshape(batch_size, channels, height, width)
        print(f'x5 size after reshaping back: {x4.size()}')

        # Decoder
        x = self.up1(x4, x3)
        print(f'size after up1: {x.size()}')
        x = self.up2(x, x2)
        print(f'size after up2: {x.size()}')
        x = self.up3(x, x1)
        print(f'size after up3: {x.size()}')
        
        return self.final_conv(x)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_dropout_rate):
        super(Attention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.proj_dropout = nn.Dropout(attention_dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output

class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim, dropout_rate):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act_fn = F.gelu
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, hidden_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.patch_embeddings = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.patch_embeddings(x)  # (B, hidden_size, n_patches_H, n_patches_W)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden_size)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Block(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, mlp_dim, dropout_rate, attention_dropout_rate):
        super(Block, self).__init__()
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size, mlp_dim, dropout_rate)
        self.attn = Attention(hidden_size, num_attention_heads, attention_dropout_rate)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x

class Encoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_attention_heads, mlp_dim, dropout_rate, attention_dropout_rate):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(hidden_size, num_attention_heads, mlp_dim, dropout_rate, attention_dropout_rate)
            self.layer.append(layer)

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded

class Transformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, hidden_size, num_layers, num_attention_heads, mlp_dim, dropout_rate, attention_dropout_rate):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_size, patch_size, in_channels, hidden_size, dropout_rate)
        self.encoder = Encoder(hidden_size, num_layers, num_attention_heads, mlp_dim, dropout_rate, attention_dropout_rate)

    def forward(self, x):
        embedding_output = self.embeddings(x)
        encoded = self.encoder(embedding_output)
        return encoded

class ReconstructionHead(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        super(ReconstructionHead, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.upsampling(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=(256, 160), patch_size=8, in_channels=2, hidden_size=768, num_layers=12, num_attention_heads=12, mlp_dim=3072, dropout_rate=0.1, attention_dropout_rate=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_size = patch_size
        self.transformer = Transformer(img_size, patch_size, in_channels, hidden_size, num_layers, num_attention_heads, mlp_dim, dropout_rate, attention_dropout_rate)
        self.reconstruction_head = ReconstructionHead(hidden_size, in_channels, kernel_size=3, upsampling=patch_size)

    def forward(self, x, reference_image):
        #print(f'x shape in visionTransformer: {x.shape}')
        #if x.size(1) == 1:
        #    x = x.repeat(1, 3, 1, 1)
        encoded = self.transformer(x)

        h_patches = x.size(2) // self.patch_size
        w_patches = x.size(3) // self.patch_size
        #print(f'h_patches: {h_patches}')
        #print(f'w_patches: {h_patches}')

        encoded = encoded.permute(0, 2, 1).contiguous().view(x.size(0), -1, h_patches, w_patches)

        reconstructed = self.reconstruction_head(encoded)

        return reconstructed


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class CustomDecoder(nn.Module):
    def __init__(self):
        super(CustomDecoder, self).__init__()
        self.fc1 = nn.Linear(104832, 1024)  # Reduce dimensionality significantly
        self.fc2 = nn.Linear(1024, 128 * 43 * 27)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: (batch, 64, 86, 54)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # Output: (batch, 32, 172, 108)
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),                       # Output: (batch, 16, 172, 108)
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=3, padding=1),                        # Output: (batch, 2, 172, 108)
            nn.Sigmoid()  # Assuming you want dataset in the range [0, 1]
        )

    def forward(self, x):
        batch_size = x.size(0)
        #print(f'Batch size: {batch_size}')
        x = x.reshape(batch_size, -1)  # Flatten the input using reshape
        #print(f'Before FC1: {x.shape}')
        x = self.fc1(x)
        #print(f'Before FC2: {x.shape}')
        x = self.fc2(x)
        x = x.reshape(batch_size, 128, 43, 27)  # Reshape using reshape
        #print(f'Before deconv-layer: {x.shape}')
        x = self.deconv_layers(x)
        return x

class TransUNet(nn.Module):
    def __init__(self, img_channels, base_channels=128 ):
        super(TransUNet, self).__init__()
        
        #self.encoder1 = ConvBlock(img_channels, base_channels)
        #self.encoder2 = ConvBlock(base_channels, base_channels * 2)
        #self.encoder3 = ConvBlock(base_channels * 2, base_channels * 4)
        #self.encoder4 = ConvBlock(base_channels * 4, base_channels * 8)
        
        #self.pool = nn.MaxPool2d(2)
        
        # Adjust dimensions to match the expected input size for the transformer
        self.transformer = VisionTransformer(img_size=(32,20), patch_size=4, in_channels=1024, hidden_size=768, num_layers=12, num_attention_heads=12, mlp_dim=3072, dropout_rate=0.1, attention_dropout_rate=0.1)
        
        #self.up1 = UpBlock(base_channels * 8, base_channels * 4)
        #self.up2 = UpBlock(base_channels * 4, base_channels * 2)
        #self.up3 = UpBlock(base_channels * 2, base_channels)

        
        #self.final_conv = nn.Conv2d(base_channels, img_channels, kernel_size=1)
        pretrained_weights = './dino_deitsmall8_pretrain_full_checkpoint.pth'
        self.extractor = ViTExtractor('dino_vits8', stride=8, model_dir=pretrained_weights, device='cuda:0')
        self.decoder = CustomDecoder()


    def forward(self, x,reference_image):
        """
        # Encoder
        x1 = self.encoder1(x)
        #print(f'after encoder1: {x1.shape}')
        x2 = self.encoder2(self.pool(x1))
        #print(f'after encoder2: {x2.shape}')
        x3 = self.encoder3(self.pool(x2))
        #print(f'after encoder3: {x3.shape}')
        x4 = self.encoder4(self.pool(x3))
        """

        # Transformer
        #x = self.transformer(x4,reference_image)
        x = torch.cat([x,x[:,1,:,:].unsqueeze(1)],dim =1)
        #print(f'Before transformer: {x.shape}')
        #x = deep_features(x, self.extractor, layer=11, facet='key', bin=False, device='cuda:0')
        with torch.no_grad():  # Disable gradient calculation for extractor
            features = self.extractor.extract_descriptors(x, layer=11, facet='key', bin=False)


        #print(f'After transformer: {features.shape}')
        x = self.decoder(features)



        """
        #print(f'size after transformer: {x3.size()}')
        # Decoder
        x = self.up1(x, x3)
        #print(f'size after up1: {x.size()}')
        x = self.up2(x, x2)
        #print(f'size after up2: {x.size()}')
        x = self.up3(x, x1)
        #print(f'size after up3: {x.size()}')
        """

        return x
    



