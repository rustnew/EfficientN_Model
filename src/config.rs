use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub image_width: usize,
    pub image_height: usize,
    pub image_channels: usize,
    pub efficientnet_variant: String,
    pub dropout_rate: f64,
    pub fc1_units: usize,
    pub fc2_units: usize,
    pub num_species_classes: usize,
    pub num_stage_classes: usize,
    pub stage_loss_lambda: f64,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub data_path: String,
    pub train_val_split: f64,
    pub use_cache: bool,
    pub num_workers: usize,
    pub grad_accum_steps: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            image_width: 128,
            image_height: 128,
            image_channels: 3,
            efficientnet_variant: "b0".to_string(),
            dropout_rate: 0.3,
            fc1_units: 1280,
            fc2_units: 512,
            num_species_classes: 5,
            num_stage_classes: 4,
            stage_loss_lambda: 0.25,
            learning_rate: 0.001,
            batch_size: 2,
            num_epochs: 10,
            data_path: "data".to_string(),
            train_val_split: 0.8,
            use_cache: true,
            num_workers: 0,
            grad_accum_steps: 2,
        }
    }
}