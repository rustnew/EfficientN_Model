// ============================================
// src/efficientnet.rs
// ============================================
use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Linear, LinearConfig, PaddingConfig2d, Relu, Sigmoid,
    },
    prelude::*,
    tensor::{backend::Backend, module::adaptive_avg_pool2d, Tensor},
};

#[derive(Module, Debug)]
pub struct SqueezeExcite<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    sigmoid: Sigmoid,
}

impl<B: Backend> SqueezeExcite<B> {
    pub fn new(channels: usize, reduction: usize, device: &B::Device) -> Self {
        let reduced_channels = channels / reduction;
        Self {
            fc1: LinearConfig::new(channels, reduced_channels).init(device),
            fc2: LinearConfig::new(reduced_channels, channels).init(device),
            sigmoid: Sigmoid::new(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let batch_size = x.dims()[0];
        let channels = x.dims()[1];

        let y = adaptive_avg_pool2d(x.clone(), [1, 1]).reshape([batch_size, channels]);
        let y = self.fc1.forward(y);
        let y = self.fc2.forward(y);
        let y = self.sigmoid.forward(y);

        x * y.reshape([batch_size, channels, 1, 1])
    }
}

#[derive(Module, Debug)]
pub struct MBConv<B: Backend> {
    expand_conv: Option<Conv2d<B>>,
    depthwise_conv: Conv2d<B>,
    squeeze_excite: Option<SqueezeExcite<B>>,
    project_conv: Conv2d<B>,
    use_residual: bool,
}

impl<B: Backend> MBConv<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        expand_ratio: f32,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        reduction: usize,
        device: &B::Device,
    ) -> Self {
        let expanded_channels = (in_channels as f32 * expand_ratio) as usize;

        let expand_conv = if (expand_ratio - 1.0).abs() > f32::EPSILON {
            Some(Conv2dConfig::new([in_channels, expanded_channels], [1, 1]).init(device))
        } else {
            None
        };

        let depthwise_conv = Conv2dConfig::new([expanded_channels, expanded_channels], kernel_size)
            .with_stride(stride)
            .with_padding(PaddingConfig2d::Same)
            .with_groups(expanded_channels)
            .init(device);

        let squeeze_excite = if reduction > 0 {
            Some(SqueezeExcite::new(expanded_channels, reduction, device))
        } else {
            None
        };

        let project_conv = Conv2dConfig::new([expanded_channels, out_channels], [1, 1]).init(device);
        let use_residual = stride == [1, 1] && in_channels == out_channels;

        Self {
            expand_conv,
            depthwise_conv,
            squeeze_excite,
            project_conv,
            use_residual,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let identity = if self.use_residual { Some(x.clone()) } else { None };

        let mut y = match &self.expand_conv {
            Some(conv) => conv.forward(x.clone()),
            None => x,
        };

        y = self.depthwise_conv.forward(y);

        if let Some(se) = &self.squeeze_excite {
            y = se.forward(y);
        }

        y = self.project_conv.forward(y);

        if let Some(id) = identity {
            y + id
        } else {
            y
        }
    }
}

#[derive(Module, Debug)]
pub struct EfficientNetB0<B: Backend> {
    conv_stem: Conv2d<B>,
    blocks: Vec<MBConv<B>>,
    conv_head: Conv2d<B>,
    relu: Relu,
    num_features: usize,
}

impl<B: Backend> EfficientNetB0<B> {
    pub fn new(device: &B::Device) -> Self {
        let blocks_config = vec![
            (1, 16, 1, 1, 3),
            (6, 24, 2, 2, 3),
            (6, 40, 2, 2, 5),
            (6, 80, 3, 2, 3),
            (6, 112, 3, 1, 5),
            (6, 192, 4, 2, 5),
            (6, 320, 1, 1, 3),
        ];

        let conv_stem = Conv2dConfig::new([3, 32], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        let mut blocks = Vec::new();
        let mut in_channels = 32;

        for (t, c, n, s, k) in blocks_config {
            let out_channels = Self::round_channels(c as f32, 1.0, 8);

            for i in 0..n {
                let stride = if i == 0 { [s, s] } else { [1, 1] };
                blocks.push(MBConv::new(in_channels, out_channels, t as f32, [k, k], stride, 4, device));
                in_channels = out_channels;
            }
        }

        let conv_head = Conv2dConfig::new([in_channels, 1280], [1, 1]).init(device);

        Self {
            conv_stem,
            blocks,
            conv_head,
            relu: Relu::new(),
            num_features: 1280,
        }
    }

    fn round_channels(channels: f32, width_multiplier: f32, divisor: usize) -> usize {
        let channels = channels * width_multiplier;
        let new_channels = (channels + divisor as f32 / 2.0).max(divisor as f32);
        ((new_channels as usize) / divisor) * divisor
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut x = self.conv_stem.forward(x);
        
        for block in &self.blocks {
            x = block.forward(x);
        }
        
        x = self.conv_head.forward(x);
        let features = self.relu.forward(x);

        let batch_size = features.dims()[0];
        let pooled = adaptive_avg_pool2d(features, [1, 1]);

        pooled.reshape([batch_size, self.num_features])
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }
}
