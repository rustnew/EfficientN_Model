// src/malaria_model.rs
use crate::{
    config::ModelConfig,
    data::MalariaBatch,
    efficientnet::EfficientNetB0,
    heads::{SpeciesHead, StageHead},
};
use burn::{
    module::Module,
    nn::Relu,
    tensor::{
        backend::{AutodiffBackend, Backend},
        loss::cross_entropy_with_logits,
        Int, Tensor,
    },
    train::{TrainOutput, TrainStep, ValidStep},
};

#[derive(Debug, Clone)]
pub struct ClassificationOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub species_output: Tensor<B, 2>,
    pub species_targets: Tensor<B, 1, Int>,
}

impl<B: Backend> burn::train::metric::ItemLazy for ClassificationOutput<B> {
    type ItemSync = Self;
    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl<B: Backend> burn::train::metric::Adaptor<burn::train::metric::LossInput<B>>
    for ClassificationOutput<B>
{
    fn adapt(&self) -> burn::train::metric::LossInput<B> {
        burn::train::metric::LossInput::new(self.loss.clone())
    }
}

impl<B: Backend> burn::train::metric::Adaptor<burn::train::metric::AccuracyInput<B>>
    for ClassificationOutput<B>
{
    fn adapt(&self) -> burn::train::metric::AccuracyInput<B> {
        let predictions = self.species_output.clone().argmax(1).float();
        burn::train::metric::AccuracyInput::new(predictions, self.species_targets.clone())
    }
}

#[derive(Module, Debug)]
pub struct MalariaEfficientNet<B: Backend> {
    efficientnet: EfficientNetB0<B>,
    species_head: SpeciesHead<B>,
    stage_head: StageHead<B>,
    relu: Relu,
    stage_loss_lambda: f32,
}

impl<B: Backend> MalariaEfficientNet<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let efficientnet = EfficientNetB0::new(device);
        let num_features = efficientnet.num_features();

        let species_head = SpeciesHead::new(
            num_features,
            config.fc1_units,
            config.num_species_classes,
            config.dropout_rate,
            device,
        );

        let stage_head = StageHead::new(
            num_features,
            config.fc2_units,
            config.num_stage_classes,
            config.dropout_rate,
            device,
        );

        Self {
            efficientnet,
            species_head,
            stage_head,
            relu: Relu::new(),
            stage_loss_lambda: config.stage_loss_lambda as f32,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let features = self.efficientnet.forward(x);
        
        let species_logits = self.species_head.forward(features.clone());
        let stage_logits = self.stage_head.forward(features);
        
        (species_logits, stage_logits)
    }

    fn compute_species_loss(
        &self,
        output: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        let num_classes = output.dims()[1];
        let targets_one_hot = targets.one_hot(num_classes).float();
        
        let loss = cross_entropy_with_logits(output, targets_one_hot);
        loss.mean().unsqueeze()
    }

    fn compute_stage_loss(&self, logits: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
        let zeros = logits.zeros_like();
        let max_val = logits.clone().max_pair(zeros);
        
        let bce_term = max_val - logits.clone() * targets.clone();
        let log_term = (logits.abs().neg().exp() + 1.0).log();
        
        let loss = bce_term + log_term;
        loss.mean().unsqueeze()
    }
}

impl<B: AutodiffBackend> TrainStep<MalariaBatch<B>, ClassificationOutput<B>> for MalariaEfficientNet<B> {
    fn step(&self, batch: MalariaBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let (species_output, stage_output) = self.forward(batch.images);
        
        let species_loss = self.compute_species_loss(species_output.clone(), batch.species.clone());
        let stage_loss = self.compute_stage_loss(stage_output, batch.stages);
        
        let loss = species_loss + stage_loss * self.stage_loss_lambda;
        
        let grads = loss.backward();

        TrainOutput::new(
            self,
            grads,
            ClassificationOutput {
                loss: loss.detach(),
                species_output: species_output.detach(),
                species_targets: batch.species,
            },
        )
    }
}

impl<B: Backend> ValidStep<MalariaBatch<B>, ClassificationOutput<B>> for MalariaEfficientNet<B> {
    fn step(&self, batch: MalariaBatch<B>) -> ClassificationOutput<B> {
        let (species_output, stage_output) = self.forward(batch.images);
        
        let species_loss = self.compute_species_loss(species_output.clone(), batch.species.clone());
        let stage_loss = self.compute_stage_loss(stage_output, batch.stages);
        
        let loss = species_loss + stage_loss * self.stage_loss_lambda;

        ClassificationOutput {
            loss: loss.detach(),
            species_output,
            species_targets: batch.species,
        }
    }
}