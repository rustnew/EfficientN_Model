mod config;
mod data;
mod efficientnet;
mod heads;
mod malaria_model;
mod training;

use crate::config::ModelConfig;
use crate::training::MalariaTrainer;
use anyhow::Result;
use burn::backend::{
    wgpu::{Wgpu, WgpuDevice},
    Autodiff,
};

type Backend = Autodiff<Wgpu<f32, i32>>;

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║     EFFICIENTNET B0 - MALARIA DETECTION              ║");
    println!("║  Multi-Task: Species + Stage Classification          ║");
    println!("╚══════════════════════════════════════════════════════╝");

    let device = WgpuDevice::default();

    // ✅ Configuration optimisée pour WGPU
    let config = ModelConfig {
        image_width: 128,
        image_height: 128,
        efficientnet_variant: "b0".to_string(),
        batch_size: 2,           // ✅ RÉDUIT pour WGPU
        num_epochs: 20,
        use_cache: true,
        num_workers: 0,          // ✅ CRITIQUE: DOIT être 0 pour WGPU
        learning_rate: 0.001,
        dropout_rate: 0.3,
        data_path: "data".to_string(),
        train_val_split: 0.8,
        grad_accum_steps: 2,     // ✅ Compense le batch_size réduit
        ..Default::default()
    };

    println!("\n⚙️ Configuration EfficientNet:");
    println!("   • Image: {}x{}", config.image_width, config.image_height);
    println!("   • EfficientNet: B0");
    println!("   • Batch size: {} (optimisé WGPU)", config.batch_size);
    println!("   • Grad accum: {} (effective batch={})", 
             config.grad_accum_steps, 
             config.batch_size * config.grad_accum_steps);
    println!("   • Dropout: {}", config.dropout_rate);
    println!("   • Époques: {}", config.num_epochs);
    println!("   • Workers: {} (WGPU single-threaded)", config.num_workers);
    println!("   • Device: {:?}\n", device);

    let trainer = MalariaTrainer::<Backend>::new(config, device);
    
    match trainer.run() {
        Ok(_) => {
            println!("✅ Programme terminé avec succès!");
            Ok(())
        }
        Err(e) => {
            eprintln!("❌ Erreur pendant l'entraînement: {}", e);
            Err(e)
        }
    }
}