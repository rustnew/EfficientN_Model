use burn::{
    nn::{Dropout, DropoutConfig, Linear, LinearConfig, Relu},
    prelude::*,
    tensor::backend::Backend,
};

#[derive(Module, Debug)]
pub struct SpeciesHead<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    dropout: Dropout,
    relu: Relu,
}

impl<B: Backend> SpeciesHead<B> {
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        num_classes: usize,
        dropout_rate: f64,
        device: &B::Device,
    ) -> Self {
        Self {
            fc1: LinearConfig::new(in_features, hidden_features).init(device),
            fc2: LinearConfig::new(hidden_features, num_classes).init(device),
            dropout: DropoutConfig::new(dropout_rate).init(),
            relu: Relu::new(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = self.relu.forward(x);
        let x = self.dropout.forward(x);
        self.fc2.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct StageHead<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    dropout: Dropout,
    relu: Relu,
}

impl<B: Backend> StageHead<B> {
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        num_stages: usize,
        dropout_rate: f64,
        device: &B::Device,
    ) -> Self {
        Self {
            fc1: LinearConfig::new(in_features, hidden_features).init(device),
            fc2: LinearConfig::new(hidden_features, num_stages).init(device),
            dropout: DropoutConfig::new(dropout_rate).init(),
            relu: Relu::new(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = self.relu.forward(x);
        let x = self.dropout.forward(x);
        self.fc2.forward(x)
    }
}
