from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class SiTMAEConfig(PretrainedConfig):

    model_type = "sit_mae"

    def __init__(
        self,
        hidden_size=768,                    # 每个patch被映射到768维的向量空间中
        num_hidden_layers=12,               # 12个Transformer块
        num_attention_heads=12,             # 编码器中注意力头的数量
        intermediate_size=3072,             # 编码器中FFN的维度
        hidden_act="gelu",                  # 编码器和池化层中的激活函数
        hidden_dropout_prob=0.0,            # 嵌入、编码器和池化层中所有全连接层的dropout概率
        attention_probs_dropout_prob=0.0,   # 注意力dropout概率
        initializer_range=0.02,             # 初始化所有权重矩阵的truncated_normal_initializer的标准差
        layer_norm_eps=1e-12,               # layer norm使用的epsilon
        max_seq_len = (4096,1),             # 需要传入最大图像尺寸,以计算pos_embed的宽度
        patch_size=(8,1),                   # patch size
        num_channels=2,                     # 通道数
        qkv_bias=True,                      # 是否对查询、键和值添加偏差
        decoder_num_attention_heads=16,     # 解码器中注意力头的数量
        decoder_hidden_size=512,            # 解码器输入维度
        decoder_num_hidden_layers=8,        # 8个Transformer块
        decoder_intermediate_size=2048,     # 解码器中FFN的维度
        mask_ratio=0.75,                    # 掩码率
        norm_pix_loss=False,                # 计算loss时是否对原图归一化
        group_images=False,
        use_fs=True,                        # 是否加入频率
        attention_type="eager",             # Attention类型
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.max_seq_len = max_seq_len 
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.group_images = group_images
        self.use_fs = use_fs
        self.attention_type = attention_type