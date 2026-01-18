use crate::{
    config::ModelConfig,
    data::{MalariaBatcher, MalariaBatch, MalariaDataset, MalariaImageItem},
    malaria_model::MalariaEfficientNet,
};
use anyhow::{anyhow, Result};
use burn::{
    data::dataloader::{DataLoader, DataLoaderBuilder},
    module::Module,
    optim::{decay::WeightDecayConfig, AdamConfig},
    record::{CompactRecorder, Recorder, RecorderError},
    tensor::backend::{AutodiffBackend, Backend},
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
};
use std::sync::Arc;

pub struct MalariaTrainer<B: AutodiffBackend> {
    config: ModelConfig,
    device: B::Device,
}

impl<B: AutodiffBackend> MalariaTrainer<B> {
    pub fn new(config: ModelConfig, device: B::Device) -> Self {
        Self { config, device }
    }

    pub fn run(&self) -> Result<()> {
        println!("🚀 Démarrage de l'entraînement");
        println!("📊 Configuration:");
        println!("   - Backend: {}", std::any::type_name::<B>());
        println!("   - Device: {:?}", self.device);
        println!("   - Image size: {}x{}", self.config.image_width, self.config.image_height);
        println!("   - Batch size: {}", self.config.batch_size);
        println!("   - Learning rate: {}", self.config.learning_rate);
        
        self.validate_config()?;
        
        let model = self.create_model();
        let (train_loader, valid_loader) = self.create_dataloaders()?;
        let optim = self.create_optimizer();
        
        self.train_model(model, optim, train_loader, valid_loader)?;
        
        Ok(())
    }
    
    fn validate_config(&self) -> Result<()> {
        if self.config.batch_size == 0 {
            return Err(anyhow!("Batch size must be > 0"));
        }
        if self.config.learning_rate <= 0.0 {
            return Err(anyhow!("Learning rate must be > 0"));
        }
        if self.config.train_val_split <= 0.0 || self.config.train_val_split >= 1.0 {
            return Err(anyhow!("Train/val split must be between 0 and 1"));
        }
        if self.config.num_epochs == 0 {
            return Err(anyhow!("Number of epochs must be > 0"));
        }
        Ok(())
    }
    
    fn create_model(&self) -> MalariaEfficientNet<B> {
        println!("🛠️  Création du modèle MalariaEfficientNet...");
        let model = MalariaEfficientNet::new(&self.config, &self.device);
        
        println!("   - Modèle créé avec ~5.20M paramètres");
        
        model
    }
    
    // ✅ CORRECTION: Types corrects pour DataLoader
    fn create_dataloaders(&self) -> Result<(
        Arc<dyn DataLoader<B, MalariaBatch<B>>>,
        Arc<dyn DataLoader<B::InnerBackend, MalariaBatch<B::InnerBackend>>>,
    )> {
        println!("📁 Chargement du dataset depuis '{}'...", self.config.data_path);
        
        let dataset = MalariaDataset::new(
            &self.config.data_path,
            self.config.image_height,
            self.config.image_width,
            self.config.use_cache,
        )?;
        
        if dataset.is_empty() {
            return Err(anyhow!("Dataset is empty"));
        }
        
        println!("   - Total d'images: {}", dataset.len());
        
        let (train_data, valid_data) = dataset.split(self.config.train_val_split as f32);
        
        println!("   - Split: {} train, {} validation", train_data.len(), valid_data.len());
        
        let batcher_train = MalariaBatcher::<B>::new(
            self.config.image_height, 
            self.config.image_width,
            self.device.clone()
        );
        
        // Pour le valid loader, on utilise le backend interne (sans autodiff)
        let device_valid = <B::InnerBackend as Backend>::Device::default();
        let batcher_valid = MalariaBatcher::<B::InnerBackend>::new(
            self.config.image_height, 
            self.config.image_width,
            device_valid
        );
        
        println!("   - Batch size: {}", self.config.batch_size);
        println!("   - Workers: {} (WGPU compatible)", self.config.num_workers);
        
        // ✅ CORRECTION: DataLoaderBuilder avec MalariaImageItem comme type I
        let train_loader = DataLoaderBuilder::<B, MalariaImageItem, MalariaBatch<B>>::new(batcher_train)
            .batch_size(self.config.batch_size)
            .shuffle(42)
            .num_workers(self.config.num_workers)
            .build(train_data);
            
        let valid_loader = DataLoaderBuilder::<B::InnerBackend, MalariaImageItem, MalariaBatch<B::InnerBackend>>::new(batcher_valid)
            .batch_size(self.config.batch_size)
            .num_workers(self.config.num_workers)
            .build(valid_data);
            
        println!("✅ DataLoaders créés avec succès");
        
        Ok((train_loader, valid_loader))
    }
    
    fn create_optimizer(&self) -> AdamConfig {
        AdamConfig::new()
            .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
            .with_beta_1(0.9)
            .with_beta_2(0.999)
    }
    
    fn train_model(
        &self,
        model: MalariaEfficientNet<B>,
        optim: AdamConfig,
        train_loader: Arc<dyn DataLoader<B, MalariaBatch<B>>>,
        valid_loader: Arc<dyn DataLoader<B::InnerBackend, MalariaBatch<B::InnerBackend>>>,
    ) -> Result<()> {
        println!("🎯 Démarrage de l'entraînement pour {} époques...", self.config.num_epochs);
        println!("   - Checkpoint: ./checkpoints");
        println!("   - Métriques: Loss, Accuracy");
        
        let start_time = std::time::Instant::now();
        
        let learner = LearnerBuilder::new("./checkpoints")
            .metric_train_numeric(AccuracyMetric::new())
            .metric_valid_numeric(AccuracyMetric::new())
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .with_file_checkpointer(CompactRecorder::new())
            .num_epochs(self.config.num_epochs)
            .grads_accumulation(self.config.grad_accum_steps)
            .summary()
            .build(model, optim.init(), self.config.learning_rate);
        
        println!("🚀 Lancement de l'entraînement...");
        let trained_model = learner.fit(train_loader, valid_loader);
        
        let duration = start_time.elapsed();
        println!("⏱️  Entraînement terminé en {:?}", duration);
        
        self.save_model(trained_model.model)
            .map_err(|e| anyhow!("Erreur de sauvegarde: {}", e))?;
        
        Ok(())
    }
    
    fn save_model<B2: Backend>(&self, model: MalariaEfficientNet<B2>) -> Result<(), RecorderError> {
        println!("💾 Sauvegarde du modèle entraîné...");
        
        let recorder = CompactRecorder::new();
        let record = model.into_record();
        
        recorder.record(record, "./checkpoints/final-model".into())?;
        
        println!("✅ Modèle sauvegardé dans: ./checkpoints/final-model");
        
        Ok(())
    }
}