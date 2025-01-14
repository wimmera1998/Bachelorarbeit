import copy
import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import normal_
import torch.nn.functional as F
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init, constant_init
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from torchvision.transforms.functional import rotate
from projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention import TemporalSelfAttention
from projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import MSDeformableAttention3D
from projects.mmdet3d_plugin.bevformer.modules.decoder import CustomMSDeformableAttention
from .builder import build_fuser, FUSERS
from typing import List
import torch.nn.init as init

@FUSERS.register_module()
class DynamicWeightSumFuser(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, width: int = 200, length: int = 100) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.length = length

        # Convolutional layer to generate fusion weights
        # Kernel size is 3x3 with padding to keep spatial dimensions consistent
        self.dynamic_conv = nn.Conv2d(256, out_channels, kernel_size=3, stride=1, padding=1)
        #keine batchnorm
        # Activation function to normalize weights to range [0, 1]
        self.sigmoid = nn.Sigmoid()

        print(f"Initialized DynamicConvFuser with input_channels: {self.in_channels}, output_channels: {self.out_channels}")

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # Assign input features to respective variables
        bev_embed, aerial_feature = inputs

        # Rescale Aerial Features to match the BEV Embed statistics
        # Calculate BEV statistics
        bev_mean = bev_embed.mean(dim=(0, 1, 2, 3), keepdim=True)  # Mean over batch, height, and width
        bev_std = bev_embed.std(dim=(0, 1, 2, 3), keepdim=True)    # Std over batch, height, and width

        # Calculate Aerial statistics
        aerial_mean = aerial_feature.mean(dim=(0,1, 2, 3), keepdim=True)
        aerial_std = aerial_feature.std(dim=(0, 1, 2, 3), keepdim=True)

        # Normalize and rescale Aerial Features
        aerial_feature_scaled = (aerial_feature - aerial_mean) / (aerial_std + 1e-6)  # Normalize
        aerial_feature_scaled = aerial_feature_scaled * bev_std + bev_mean           # Rescale

        # Generate fusion weights using a convolutional layer on BEV Embed
        conv_fusion_weights = self.dynamic_conv(bev_embed)  # Output shape: [batch, out_channels, width, length]
        conv_fusion_weights = self.sigmoid(conv_fusion_weights)   # Normalize to range [0, 1]

        # Perform the fusion
        weighted_bev_embed = conv_fusion_weights * bev_embed
        weighted_aerial_feature = (1 - conv_fusion_weights) * aerial_feature_scaled

        # Calculate the fused feature
        fused_feature = weighted_bev_embed + weighted_aerial_feature
      

        return fused_feature


@FUSERS.register_module()
class AdvancedDynamicConvFuser(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, width: int = 200, length: int = 100) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.length = length

        # Convolutional layer to generate initial fusion weights
        self.initial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()


        # Convolutional layer to combine weighted features
        self.combine_conv = nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        #batch norm und aktivu
        #nn.BatchNorm2d(out_channels),
        #nn.ReLU(True),


        print(f"Initialized AdvancedDynamicConvFuser with input_channels: {self.in_channels}, output_channels: {self.out_channels}")

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # Assign input features to respective variables
        bev_embed, aerial_feature = inputs

        # Calculate BEV statistics and rescale aerial features
        bev_mean = bev_embed.mean(dim=(0, 1, 2, 3), keepdim=True)
        bev_std = bev_embed.std(dim=(0, 1, 2, 3), keepdim=True)
        aerial_mean = aerial_feature.mean(dim=(0, 1, 2, 3), keepdim=True)
        aerial_std = aerial_feature.std(dim=(0, 1, 2, 3), keepdim=True)
        aerial_feature_scaled = (aerial_feature - aerial_mean) / (aerial_std + 1e-6)
        aerial_feature_scaled = aerial_feature_scaled * bev_std + bev_mean

        # Generate initial fusion weights
        conv_fusion_weights = self.initial_conv(bev_embed)
        conv_fusion_weights = self.sigmoid(conv_fusion_weights)

        # Apply weights to BEV and aerial features
        weighted_bev_embed = conv_fusion_weights * bev_embed
        weighted_aerial_feature = (1 - conv_fusion_weights) * aerial_feature_scaled

        # Concatenate weighted features along channel dimension
        combined_features = torch.cat([weighted_bev_embed, weighted_aerial_feature], dim=1)

        # Use convolution to fuse combined features
        fused_feature = self.combine_conv(combined_features)

        return fused_feature




class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, height, width):
        super(PositionalEncoding2D, self).__init__()
        self.channels = channels // 2 * 2  # ensure even number of channels
        self.height = height
        self.width = width
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.channels, 2).float() / self.channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        device = tensor.device
        pos_x = torch.arange(self.height, device=device).float()
        pos_y = torch.arange(self.width, device=device).float()
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq).view(self.height, 1, -1)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq).view(1, self.width, -1)
        pos_emb = torch.cat((torch.sin(sin_inp_x), torch.cos(sin_inp_x), torch.sin(sin_inp_y), torch.cos(sin_inp_y)), dim=-1)
        pos_emb = pos_emb.permute(2, 0, 1).unsqueeze(0)
        pos_emb = pos_emb.repeat(tensor.shape[0], 1, 1, 1)
        return pos_emb

class ConvFuser(nn.Module):
    def __init__(self, in_channels, out_channels, width, height, num_heads):
        super().__init__()
        self.pos_enc = PositionalEncoding2D(in_channels, width, height)
        self.conv_q = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_k = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_v = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.multihead_attn = nn.MultiheadAttention(out_channels, num_heads)

    def forward(self, bev_embed, aerial_feature):
        # Add positional encoding
        bev_embed_pos_enc = self.pos_enc(bev_embed)
        aerial_feature_pos_enc = self.pos_enc(aerial_feature)

        # Generate queries, keys, values
        q = self.conv_q(aerial_feature_pos_enc)
        k = self.conv_k(bev_embed_pos_enc)
        v = self.conv_v(bev_embed_pos_enc)

        # Reshape and permute for multihead attention
        batch_size = q.size(0)
        q = q.view(batch_size, self.out_channels, -1)
        k = k.view(batch_size, self.out_channels, -1)
        v = v.view(batch_size, self.out_channels, -1)

        # Apply multihead attention
        attn_output, _ = self.multihead_attn(q, k, v)
        attn_output = attn_output.view(batch_size, self.out_channels, self.width, self.height)

        return attn_output











#cwl_fusion
@FUSERS.register_module()
class CWLFuser(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, width: int = 200, length: int = 100) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.length = length

        # Define a learnable fusion weight tensor with shape (out_channels, width, length)
        # Xavier initialization to keep the weights in a suitable range
        self.fusion_weight_tensor = nn.Parameter(torch.empty((out_channels, width, length)))
        init.xavier_uniform_(self.fusion_weight_tensor)  # Apply Xavier initialization

        # BatchNorm for Aerial Features
        #self.aerial_bn = nn.BatchNorm2d(num_features=in_channels)

        print(f"Initialized CWLFuser with Xavier-initialized fusion weight tensor of shape {self.fusion_weight_tensor.shape}")
        print(f"in_channels: {self.in_channels}, out_channels: {self.out_channels}")
        print("F1")

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # Assign input features to respective variables for clarity
        bev_embed, aerial_feature = inputs


        # Calculate BEV statistics
        bev_mean = bev_embed.mean(dim=(0, 1, 2, 3), keepdim=True)  # Mean over batch, height, and width
        bev_std = bev_embed.std(dim=(0, 1, 2, 3), keepdim=True)    # Std over batch, height, and width

        # Calculate Aerial statistics
        aerial_mean = aerial_feature.mean(dim=(0, 1, 2, 3), keepdim=True)
        aerial_std = aerial_feature.std(dim=(0, 1, 2, 3), keepdim=True)

        # Normalize and rescale Aerial Features
        aerial_feature_scaled = (aerial_feature - aerial_mean) / (aerial_std + 1e-6)  # Normalize 1e−6 wird hinzugefügt, um Division durch Null zu vermeiden, falls die Standardabweichung sehr klein ist.
        aerial_feature_scaled = aerial_feature_scaled * bev_std + bev_mean  # Rescale to BEV's stats
        
        # Expand `fusion_weight_tensor` to match the shape of `bev_embed` and `aerial_feature` (batch dimension added)
        fusion_weight_map = self.fusion_weight_tensor.view(1, self.out_channels, self.width, self.length)
                
        # Perform weighted fusion
        bev_weighted_features = fusion_weight_map * bev_embed
        aerial_weighted_features = (1 - fusion_weight_map) * aerial_feature_scaled
        fused_feature = bev_weighted_features + aerial_weighted_features

        # Debugging information
        #print(f"BEV Embed Mean: {bev_mean.flatten().mean().item():.4f}, Std: {bev_std.flatten().mean().item():.4f}")
        #print(f"Aerial Features Mean (before scaling): {aerial_mean.flatten().mean().item():.4f}, Std: {aerial_std.flatten().mean().item():.4f}")
        #print(f"Aerial Features Mean (after scaling): {aerial_feature_scaled.mean().item():.4f}, Std: {aerial_feature_scaled.std().item():.4f}")

      

        #print("G5")         
        return fused_feature












'''
@FUSERS.register_module()
class ConvFuser(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int) -> None:
        super().__init__()
        
        # Speichern der Kanäle
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Aufbau des Fusionsmoduls als nn.Sequential
        self.fusion_layers = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        print("Fusion_Layer")
        for layer in self.fusion_layers:
            print(layer)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        print("Starting forward pass in ConvFuser")

        # Konkatenation der Eingaben entlang der Kanal-Dimension
        concatenated_input = torch.cat(inputs, dim=1)
        print(f"Concatenated Input Shape: {concatenated_input.shape}")

        # Führen Sie die Fusion durch nn.Sequential aus
        out = self.fusion_layers(concatenated_input)
        
        print(f"Output Shape after ConvFuser: {out.shape}")
        
        return out

'''





'''
by Julius
@FUSERS.register_module()
class ConvFuser(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        print("Hallo")
        print("self.in_channels", self.in_channels)
        print("self.out_channels", self.out_channels)



        self.conv = nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False)
        print(self.conv)

        self.bn = nn.BatchNorm2d(sum(in_channels))

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        print("Start forward")
        mask_inp = self.bn(torch.cat(inputs, dim=1))
        print("Mask input:", mask_inp.mean(), mask_inp.std())
        mask = self.conv(mask_inp)
        mask = F.sigmoid(mask)
        print('Mask:', mask.mean(), mask.std())

        out = inputs[0] * mask + inputs[1] * (1 - mask)

        print("ConvFuser output:", out.shape)
        return out
'''

@FUSERS.register_module()
class BEVOnlyFuser(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        # Speichern der Kanäle für das BEV-Feature und die Zielausgabe
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initialisierung der Convolutional Layer, BatchNorm und ReLU
        #self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        #self.bn = nn.BatchNorm2d(out_channels)
        #self.relu = nn.ReLU(True)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # Nur das BEV-Feature verwenden (das erste Feature in der Liste)
        bev_embed = inputs[0]  # Das BEV-Feature (Feature 0)

        # Durch den Convolutional Layer, BatchNorm und ReLU führen
        #bev_embed = self.conv(bev_embed)
        #bev_embed = self.bn(bev_embed)
        #bev_embed = self.relu(bev_embed)

        return bev_embed





@TRANSFORMER.register_module()
class MapTRPerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 fuser=None,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 len_can_bus=18,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 modality='vision',
                 **kwargs):
        super(MapTRPerceptionTransformer, self).__init__(**kwargs)
        if modality == 'fusion':
            self.fuser = build_fuser(fuser) #TODO
        fuser = {'type': 'CWLFuser', 'in_channels': [256, 256], 'out_channels': 256}
        self.fuser = build_fuser(fuser)
        self.use_attn_bev = encoder['type'] == 'BEVFormerEncoder'
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.len_can_bus = len_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center
        print("init wird aufgerufen")

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 2) # TODO, this is a hack
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(self.len_can_bus, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))
        print("init_layer wird aufgerufen")


    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)
    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'), out_fp32=True)
    def attn_bev_encode(
            self,
            mlvl_feats,
            # aerial_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus[:, :self.len_can_bus])[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            # aerial_feats,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs
        )
        return bev_embed

    def lss_bev_encode(
            self,
            mlvl_feats,
            prev_bev=None,
            **kwargs):
        assert len(mlvl_feats) == 1, 'Currently we only support single level feat in LSS'
        images = mlvl_feats[0]
        img_metas = kwargs['img_metas']
        bev_embed = self.encoder(images,img_metas)
        bs, c, _,_ = bev_embed.shape
        bev_embed = bev_embed.view(bs,c,-1).permute(0,2,1).contiguous()
        
        return bev_embed

    def get_bev_features(
            self,
            mlvl_feats,
            aerial_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """
        if self.use_attn_bev:
            bev_embed = self.attn_bev_encode(
                mlvl_feats,
                #aerial_feats,
                bev_queries,
                bev_h,
                bev_w,
                grid_length=grid_length,
                bev_pos=bev_pos,
                prev_bev=prev_bev,
                **kwargs)
            bs = mlvl_feats[0].size(0)
            bev_embed = bev_embed.view(bs, bev_h, bev_w, -1).permute(0,3,1,2).contiguous()
            aerial_feats = nn.functional.interpolate(aerial_feats, size=(bev_h,bev_w), mode='bicubic', align_corners=True)
            
            # Zeige die Herkunft der Features an
            print("Anfang in: get_bev_features")
            print("Feature: BEV Embed")
            print(f"Shape: {bev_embed.shape}")
            print(f"Mean: {bev_embed.mean().item():.4f}, Std: {bev_embed.std().item():.4f}")
            print(f"Min: {bev_embed.min().item():.4f}, Max: {bev_embed.max().item():.4f}")

            # aerial_feats interpolieren
            print("Feature: Aerial Features")
            print(f"Shape: {aerial_feats.shape}")
            print(f"Mean: {aerial_feats.mean().item():.4f}, Std: {aerial_feats.std().item():.4f}")
            print(f"Min: {aerial_feats.min().item():.4f}, Max: {aerial_feats.max().item():.4f}")
            print("Ende in: get_bev_features")
            
            
            
            
            
            
            
            
            fused_bev = self.fuser([bev_embed, aerial_feats])
            fused_bev = fused_bev.flatten(2).permute(0,2,1).contiguous()
            bev_embed = fused_bev



        else:
            bev_embed = self.lss_bev_encode(
                mlvl_feats,
                prev_bev=prev_bev,
                **kwargs)
        if lidar_feat is not None:
            bs = mlvl_feats[0].size(0)
            bev_embed = bev_embed.view(bs, bev_h, bev_w, -1).permute(0,3,1,2).contiguous()
            lidar_feat = lidar_feat.permute(0,1,3,2).contiguous() # B C H W
            lidar_feat = nn.functional.interpolate(lidar_feat, size=(bev_h,bev_w), mode='bicubic', align_corners=False)
            fused_bev = self.fuser([bev_embed, lidar_feat])
            fused_bev = fused_bev.flatten(2).permute(0,2,1).contiguous()
            bev_embed = fused_bev
            print("Feature 3: Lidar Features")
            print(f"Shape: {lidar_feat.shape}")

        return bev_embed
    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                aerial_feats,
                lidar_feat,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        bev_embed = self.get_bev_features(
            mlvl_feats,
            aerial_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out
