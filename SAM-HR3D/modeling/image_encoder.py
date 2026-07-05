import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type
from segment_anything.modeling.common import LayerNorm2d, MLPBlock
from segment_anything.modeling.image_encoder import Attention, PatchEmbed, window_partition, window_unpartition


# 优化后的SCR模块
class SCR(nn.Module):
    def __init__(self, dim: int, reduction_ratio: int = 4):
        super().__init__()
        self.dim = dim
        reduced_dim = dim // reduction_ratio
        self.spatial_attn = nn.Sequential(
            nn.Conv3d(dim, reduced_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(reduced_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced_dim, 1, kernel_size=1, bias=True),
            nn.Softmax(dim=-1)
        )
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dim, reduced_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced_dim, dim, kernel_size=1, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, D, C = x.shape
        assert C == self.dim, f"Expected channel dimension {self.dim}, got {C}"
        x_in = x
        x_perm = x.permute(0, 4, 1, 2, 3)
        spatial_weight = self.spatial_attn(x_perm).permute(0, 2, 3, 4, 1)
        spatial_out = x * spatial_weight
        channel_weight = self.channel_attn(x_perm).permute(0, 2, 3, 4, 1)
        channel_out = x * channel_weight
        out = spatial_out + channel_out + x_in
        return out


class Adapter(nn.Module):
    def __init__(self, input_dim: int, mid_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, mid_dim)
        self.conv_1x1 = nn.Conv3d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=1, padding=0, groups=mid_dim)
        self.conv_3x3 = nn.Conv3d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, padding=1, groups=mid_dim)
        self.conv_5x5 = nn.Conv3d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=5, padding=2, groups=mid_dim)
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)
        self.linear2 = nn.Linear(mid_dim, input_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.linear1(features)
        out = F.relu(out)
        out = out.permute(0, 4, 1, 2, 3)
        out_1x1 = self.conv_1x1(out)
        out_3x3 = self.conv_3x3(out)
        out_5x5 = self.conv_5x5(out)
        scale_weights = F.softmax(self.scale_weights, dim=0)
        out = scale_weights[0] * out_1x1 + scale_weights[1] * out_3x3 + scale_weights[2] * out_5x5
        out = out.permute(0, 2, 3, 4, 1)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = features + out
        return out


class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class Attention_3d(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: Optional[Tuple[int, int, int]] = None,
            res_size: Optional[Tuple[int, int, int]] = None
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * res_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * res_size[1] - 1, head_dim))
            self.rel_pos_d = nn.Parameter(torch.zeros(2 * res_size[2] - 1, head_dim))
            self.lr = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, H, W, D, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W * D, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q_sub = q.reshape(B * self.num_heads, H * W * D, -1)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q_sub, self.rel_pos_h, self.rel_pos_w, self.rel_pos_d, (H, W, D),
                                          (H, W, D), self.lr)
            attn = attn.reshape(B, self.num_heads, H * W * D, -1)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, H * W * D, H * W * D) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, H * W * D, H * W * D)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, D, -1).permute(0, 2, 3, 4, 1, 5).reshape(B, H, W, D, -1)
        x = self.proj(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    B, H, W, D, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    pad_d = (window_size - D % window_size) % window_size
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        x = F.pad(x, (0, 0, 0, pad_d, 0, pad_w, 0, pad_h))
    Hp, Wp, Dp = H + pad_h, W + pad_w, D + pad_d
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, Dp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows, (Hp, Wp, Dp)


def window_unpartition(windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int, int],
                       hw: Tuple[int, int, int]) -> torch.Tensor:
    Hp, Wp, Dp = pad_hw
    H, W, D = hw
    B = windows.shape[0] // (Hp * Wp * Dp // window_size // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, Dp // window_size, window_size, window_size, window_size,
                     -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Hp, Wp, Dp, -1)
    if Hp > H or Wp > W or Dp > D:
        x = x[:, :H, :W, :D, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
        attn: torch.Tensor,
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        rel_pos_d: torch.Tensor,
        q_size: Tuple[int, int, int],
        k_size: Tuple[int, int, int],
        lr: torch.Tensor,
) -> torch.Tensor:
    q_h, q_w, q_d = q_size
    k_h, k_w, k_d = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    Rd = get_rel_pos(q_d, k_d, rel_pos_d)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, q_d, dim)
    rel_h = torch.einsum("bhwdc,hkc->bhwdk", r_q, Rh)
    rel_w = torch.einsum("bhwdc,wkc->bhwdk", r_q, Rw)
    rel_d = torch.einsum("bhwdc,dkc->bhwdk", r_q, Rd)
    attn = (
            attn.view(B, q_h, q_w, q_d, k_h, k_w, k_d) +
            lr * rel_h[:, :, :, :, :, None, None] +
            lr * rel_w[:, :, :, :, None, :, None] +
            lr * rel_d[:, :, :, :, None, None, :]
    ).view(B, q_h * q_w * q_d, k_h * k_w * k_d)
    return attn


class Block_3d(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            res_size: Optional[int] = None,
            shift: Optional[int] = None,
    ):
        super().__init__()
        self.window_size = window_size
        logger = logging.getLogger(__name__)
        logger.info(f"Block_3d initialized with window_size={window_size}, shift={shift}")
        self.norm1 = norm_layer(dim)
        self.attn = Attention_3d(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=(window_size, window_size, window_size) if window_size > 0 else None,
            res_size=(res_size, res_size, res_size) if res_size is not None else None,
        )
        self.shift_size = shift or 0
        if self.shift_size > 0 and window_size > 0:
            H, W, D = 32, 32, 32
            img_mask = torch.zeros((1, H, W, D, 1))
            h_slices = (slice(0, -window_size), slice(-window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -window_size), slice(-window_size, -self.shift_size), slice(-self.shift_size, None))
            d_slices = (slice(0, -window_size), slice(-window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    for d in d_slices:
                        img_mask[:, h, w, d, :] = cnt
                        cnt += 1
            mask_windows, _ = window_partition(img_mask, window_size)
            mask_windows = mask_windows.view(-1, window_size * window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.adapter = Adapter(input_dim=dim, mid_dim=dim // 4)

        # ===================== 修改位置 1/3 =====================
        # 将投影维度从64改为128，并将ipf改为SPF
        self.SPF_projection1 = nn.Linear(dim, 128)  # 修改前: self.ipf_projection1
        # 更新SCR的维度
        self.SPF_SCR = SCR(dim=128, reduction_ratio=4)  # 修改前: self.ipf_triplet_attn
        self.SPF_projection2 = nn.Linear(128, dim)  # 修改前: self.ipf_projection2
        # ===================== 修改结束 =====================

    def forward(self, x: torch.Tensor, x_int: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)

        # ===================== 修改位置 2/3 =====================
        # 投影维度已修改为128，并将ipf改为SPF
        fused_features = self.SPF_projection1(x) + self.SPF_projection1(x_int)
        fused_features = self.SPF_SCR(fused_features)
        x = self.SPF_projection2(fused_features)
        # ===================== 修改结束 =====================

        if self.window_size > 0:
            H, W, D = x.shape[1], x.shape[2], x.shape[3]
            if self.shift_size > 0:
                x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
            x, pad_hw = window_partition(x, self.window_size)
            x = self.attn(x, mask=self.attn_mask)
            x = window_unpartition(x, self.window_size, pad_hw, (H, W, D))
            if self.shift_size > 0:
                x = torch.roll(x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
        else:
            x = self.attn(x, mask=None)
        x = shortcut + x
        x = self.norm2(x)
        x = x + self.mlp(x)
        x = self.adapter(x)
        return x


class ImageEncoderViT_3d_v2(nn.Module):
    def __init__(
            self,
            img_size: int = 1024,
            patch_size: int = 16,
            patch_depth: int = 32,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            cubic_window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
            num_slice: int = 1
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.num_slice = num_slice
        if self.num_slice > 1:
            self.slice_embed = nn.Conv3d(
                in_channels=embed_dim, out_channels=embed_dim,
                kernel_size=(1, 1, self.num_slice), stride=(1, 1, self.num_slice),
                groups=embed_dim
            )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
        ) if use_abs_pos else None
        self.depth_embed = nn.Parameter(
            torch.ones(1, patch_depth, embed_dim)
        ) if use_abs_pos else None
        self.shared_projection = nn.Linear(embed_dim, embed_dim)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block_3d(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=cubic_window_size,
                res_size=window_size if i not in global_attn_indexes else img_size // patch_size,
                shift=cubic_window_size // 2 if i % 2 == 0 else 0
            )
            self.blocks.append(block)
        self.neck_3d = nn.ModuleList()
        for i in range(4):
            self.neck_3d.append(nn.Sequential(
                nn.Conv3d(embed_dim, out_chans, 1, bias=False),
                LayerNorm3d(out_chans),
                nn.Conv3d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
                LayerNorm3d(out_chans),
            ))
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        if self.num_slice > 1:
            for param in self.slice_embed.parameters():
                param.requires_grad = False
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
            self.depth_embed.requires_grad = False
        for block in self.blocks:
            for name, param in block.named_parameters():
                if "SPF" in name or "adapter" in name or "shared_projection" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        for neck in self.neck_3d:
            for param in neck.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        x = self.patch_embed(x)
        x_0 = x
        x_int = self.shared_projection(x_0)
        if self.num_slice > 1:
            x = self.slice_embed(x.permute(3, 1, 2, 0).unsqueeze(0))
            x = x.permute(0, 2, 3, 4, 1)
            x_int = self.slice_embed(x_int.permute(3, 1, 2, 0).unsqueeze(0))
            x_int = x_int.permute(0, 2, 3, 4, 1)
        else:
            x = x.permute(1, 2, 0, 3).unsqueeze(0)
            x_int = x_int.permute(1, 2, 0, 3).unsqueeze(0)
        if self.pos_embed is not None:
            pos_embed = F.avg_pool2d(self.pos_embed.permute(0, 3, 1, 2), kernel_size=2).permute(0, 2, 3, 1).unsqueeze(3)
            pos_embed = pos_embed + self.depth_embed.unsqueeze(1).unsqueeze(1)
            x = x + pos_embed
            x_int = x_int + pos_embed
        feature_list = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x, x_int)
            if idx % 3 == 0 and idx != 12:
                feature_list.append(self.neck_3d[idx // 3](x.permute(0, 4, 1, 2, 3)))
        x = self.neck_3d[-1](x.permute(0, 4, 1, 2, 3))
        return x, feature_list


