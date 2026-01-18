use anyhow::{anyhow, Result};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{backend::Backend, Int, Tensor},
};
use image::{imageops::FilterType, ImageReader};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use rayon::prelude::*;
use std::{
    fs,
    path::{Path, PathBuf},
};

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug, Clone, PartialEq)]
pub struct ImageLabels {
    pub species: u8,
    pub stages: [u8; 4],
    pub infected: u8,
}

#[derive(Debug, Clone)]
pub struct MalariaImageItem {
    pub image_path: PathBuf,
    pub labels: ImageLabels,
}

#[derive(Debug, Clone)]
pub struct MalariaDataset {
    pub items: Vec<MalariaImageItem>,
    pub target_height: usize,
    pub target_width: usize,
}

impl MalariaDataset {
    pub fn new<P: AsRef<Path>>(
        data_dir: P,
        target_height: usize,
        target_width: usize,
        _use_cache: bool,
    ) -> Result<Self> {
        let data_dir = data_dir.as_ref();
        println!("📂 Chargement des données depuis: {}", data_dir.display());

        let image_files = Self::collect_image_files(data_dir)?;
        println!("📊 Nombre total d'images trouvées: {}", image_files.len());

        let items: Vec<MalariaImageItem> = image_files
            .par_iter()
            .filter_map(|path| {
                Self::parse_labels_from_path(path)
                    .map_err(|e| {
                        eprintln!("⚠️  Impossible de parser {}: {}", path.display(), e);
                    })
                    .ok()
            })
            .collect();

        if items.is_empty() {
            return Err(anyhow!("Aucune image avec des labels valides trouvée!"));
        }

        let mut items = items;
        let mut rng = StdRng::seed_from_u64(42);
        items.shuffle(&mut rng);

        Self::print_statistics(&items);

        Ok(Self {
            items,
            target_height,
            target_width,
        })
    }

    fn collect_image_files(dir: &Path) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        let mut dirs = vec![dir.to_path_buf()];
        let valid_extensions = ["png", "jpg", "jpeg", "tif", "tiff", "bmp"];

        while let Some(current_dir) = dirs.pop() {
            let entries = fs::read_dir(&current_dir)
                .map_err(|e| anyhow!("Erreur lecture dossier {}: {}", current_dir.display(), e))?;

            for entry in entries {
                let entry = entry?;
                let path = entry.path();
                
                if path.is_dir() {
                    dirs.push(path);
                } else if path.is_file() {
                    if let Some(ext) = path.extension() {
                        let ext = ext.to_string_lossy().to_lowercase();
                        if valid_extensions.iter().any(|&e| e == ext) {
                            files.push(path);
                        }
                    }
                }
            }
        }
        Ok(files)
    }

    fn parse_labels_from_path(path: &Path) -> Result<MalariaImageItem> {
        let labels = Self::parse_labels_from_filename(path)?;
        
        Ok(MalariaImageItem {
            image_path: path.to_path_buf(),
            labels,
        })
    }

    fn parse_labels_from_filename(path: &Path) -> Result<ImageLabels> {
        let filename = path.file_stem()
            .ok_or_else(|| anyhow!("Nom de fichier invalide: {}", path.display()))?
            .to_string_lossy()
            .to_lowercase();
        
        let species = if filename.contains("uninfected") || filename.contains("non-infected") {
            4
        } else if filename.contains("falciparum") && !filename.contains("non-falciparum") {
            0
        } else if filename.contains("malariae") {
            1
        } else if filename.contains("ovale") {
            2
        } else if filename.contains("vivax") {
            3
        } else {
            0
        };
        
        let mut stages = [0u8; 4];
        let filename_upper = filename.to_uppercase();
        
        if filename_upper.contains("-R") || filename.contains("ring") {
            stages[0] = 1;
        }
        if filename_upper.contains("-T") || filename.contains("troph") {
            stages[1] = 1;
        }
        if filename_upper.contains("-S") || filename.contains("schizont") {
            stages[2] = 1;
        }
        if filename_upper.contains("-G") || filename.contains("gameto") {
            stages[3] = 1;
        }
        
        let infected = if species == 4 { 0 } else { 1 };
        if infected == 1 && stages.iter().all(|&s| s == 0) {
            stages = [1, 1, 1, 1];
        }
        
        Ok(ImageLabels {
            species,
            stages,
            infected,
        })
    }

    fn print_statistics(items: &[MalariaImageItem]) {
        let total = items.len();
        let infected: usize = items.iter().filter(|i| i.labels.infected == 1).count();
        let uninfected = total - infected;
        
        let mut species_counts = [0; 5];
        for item in items {
            species_counts[item.labels.species as usize] += 1;
        }
        
        let mut stage_counts = [0; 4];
        for item in items {
            for i in 0..4 {
                if item.labels.stages[i] == 1 {
                    stage_counts[i] += 1;
                }
            }
        }
        
        println!("📊 Statistiques du dataset:");
        println!("   • Total: {} images", total);
        println!("   • Infecté: {} ({}%)", infected, infected * 100 / total.max(1));
        println!("   • Non-infecté: {} ({}%)", uninfected, uninfected * 100 / total.max(1));
        println!("   • Espèces:");
        let species_names = ["Falciparum", "Malariae", "Ovale", "Vivax", "Uninfected"];
        for (i, name) in species_names.iter().enumerate() {
            println!("     - {}: {}", name, species_counts[i]);
        }
        println!("   • Stades (multi-label):");
        let stage_names = ["Ring (R)", "Trophozoite (T)", "Schizont (S)", "Gametocyte (G)"];
        for (i, name) in stage_names.iter().enumerate() {
            println!("     - {}: {}", name, stage_counts[i]);
        }
    }

    pub fn load_and_preprocess_image(
        path: &Path,
        target_height: usize,
        target_width: usize,
    ) -> Result<Vec<f32>> {
        let img = ImageReader::open(path)?.decode()?.resize_exact(
            target_width as u32,
            target_height as u32,
            FilterType::Triangle,
        );

        let rgb_img = img.to_rgb8();
        let raw_pixels = rgb_img.into_raw();
        let frame_size = target_height * target_width;
        
        let mut chw_data = vec![0.0; frame_size * 3];
        
        for i in 0..frame_size {
            let base = i * 3;
            chw_data[i] = (raw_pixels[base] as f32 / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
            chw_data[i + frame_size] = (raw_pixels[base + 1] as f32 / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
            chw_data[i + 2 * frame_size] = (raw_pixels[base + 2] as f32 / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
        }
        
        Ok(chw_data)
    }

    pub fn split(&self, ratio: f32) -> (Self, Self) {
        assert!(ratio > 0.0 && ratio < 1.0, "Ratio doit être entre 0 et 1");
        let split_index = (self.items.len() as f32 * ratio) as usize;

        let train_items = self.items[..split_index].to_vec();
        let valid_items = self.items[split_index..].to_vec();

        println!("📈 Split du dataset (ratio: {}):", ratio);
        println!("   - Entraînement: {} images", train_items.len());
        println!("   - Validation: {} images", valid_items.len());

        (
            Self {
                items: train_items,
                target_height: self.target_height,
                target_width: self.target_width,
            },
            Self {
                items: valid_items,
                target_height: self.target_height,
                target_width: self.target_width,
            },
        )
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<MalariaImageItem> {
        self.items.get(index).cloned()
    }
}

impl Dataset<MalariaImageItem> for MalariaDataset {
    fn get(&self, index: usize) -> Option<MalariaImageItem> {
        self.get(index)
    }

    fn len(&self) -> usize {
        self.len()
    }
}

#[derive(Debug, Clone)]
pub struct MalariaBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub species: Tensor<B, 1, Int>,
    pub stages: Tensor<B, 2>,
    pub infected: Tensor<B, 1, Int>,
}

#[derive(Debug, Clone)]
pub struct MalariaBatcher<B: Backend> {
    pub image_height: usize,
    pub image_width: usize,
    pub device: B::Device,
}

impl<B: Backend> MalariaBatcher<B> {
    pub fn new(image_height: usize, image_width: usize, device: B::Device) -> Self {
        Self {
            image_height,
            image_width,
            device,
        }
    }
}

// ✅ CORRECTION: Méthode batch avec 3 paramètres
impl<B: Backend> Batcher<B, MalariaImageItem, MalariaBatch<B>> for MalariaBatcher<B> {
    fn batch(&self, items: Vec<MalariaImageItem>, device: &B::Device) -> MalariaBatch<B> {
        let batch_size = items.len();
        let frame_size = self.image_height * self.image_width;
        
        let mut images_data = Vec::with_capacity(batch_size * 3 * frame_size);
        let mut species_data = Vec::with_capacity(batch_size);
        let mut stages_data = Vec::with_capacity(batch_size * 4);
        let mut infected_data = Vec::with_capacity(batch_size);
        
        for item in &items {
            match MalariaDataset::load_and_preprocess_image(
                &item.image_path,
                self.image_height,
                self.image_width,
            ) {
                Ok(data) => {
                    images_data.extend_from_slice(&data);
                    species_data.push(item.labels.species as i64);
                    infected_data.push(item.labels.infected as i64);
                    
                    for stage in &item.labels.stages {
                        stages_data.push(*stage as f32);
                    }
                }
                Err(e) => {
                    eprintln!("⚠️ Erreur chargement {}: {}", item.image_path.display(), e);
                    images_data.extend(vec![0.0; 3 * frame_size]);
                    species_data.push(0);
                    infected_data.push(0);
                    stages_data.extend(vec![0.0; 4]);
                }
            }
        }
        
        // Utilisation de &* pour convertir &Vec<T> en &[T]
        let images_tensor = Tensor::<B, 1>::from_floats(&*images_data, device)
            .reshape([batch_size as i32, 3, self.image_height as i32, self.image_width as i32]);
        
        let species_tensor = Tensor::<B, 1, Int>::from_ints(&*species_data, device);
        let stages_tensor = Tensor::<B, 1>::from_floats(&*stages_data, device)
            .reshape([batch_size as i32, 4]);
        let infected_tensor = Tensor::<B, 1, Int>::from_ints(&*infected_data, device);
        
        MalariaBatch {
            images: images_tensor,
            species: species_tensor,
            stages: stages_tensor,
            infected: infected_tensor,
        }
    }
}