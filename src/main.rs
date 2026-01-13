use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize, Debug)]
struct Size {
    height: u32,
    width: u32,
}

#[derive(Serialize, Deserialize, Debug)]
struct Bitmap {
    origin: [u32; 2],
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
}

#[derive(Serialize, Deserialize, Debug)]
struct Object {
    classTitle: String,
    geometryType: String,
    bitmap: Bitmap,
}

#[derive(Serialize, Deserialize, Debug)]
struct CleanedJson {
    size: Size,
    object: Object,
}

fn process_folder(input_folder: &Path, output_folder: &Path) -> std::io::Result<()> {
    // Créer le dossier output s'il n'existe pas
    if !output_folder.exists() {
        fs::create_dir(output_folder)?;
    }

    for entry in fs::read_dir(input_folder)? {
        let entry = entry?;
        let path = entry.path();

        // Si c'est un dossier, on le saute (pour l'instant)
        if path.is_dir() {
            continue;
        }

        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            println!("🔹 Traitement du fichier {:?}", path.file_name().unwrap());

            let file = File::open(&path)?;
            let reader = BufReader::new(file);
            let json_value: Value = match serde_json::from_reader(reader) {
                Ok(v) => v,
                Err(_) => {
                    println!("⚠️  Impossible de lire {:?}", path);
                    continue;
                }
            };

            // Sauter si JSON vide ou sans objet
            if json_value.is_null() || json_value["objects"].as_array().unwrap_or(&vec![]).is_empty() {
                println!("⚠️  Fichier vide ou sans objet, on saute: {:?}", path.file_name().unwrap());
                continue;
            }

            // Prendre le premier objet seulement
            let first_object = &json_value["objects"][0];

            // Construire le JSON nettoyé
            let cleaned = CleanedJson {
                size: serde_json::from_value(json_value["size"].clone())
                    .unwrap_or(Size { height: 0, width: 0 }),
                object: Object {
                    classTitle: first_object["classTitle"].as_str().unwrap_or("").to_string(),
                    geometryType: first_object["geometryType"].as_str().unwrap_or("").to_string(),
                    bitmap: Bitmap {
                        origin: [
                            first_object["bitmap"]["origin"][0].as_u64().unwrap_or(0) as u32,
                            first_object["bitmap"]["origin"][1].as_u64().unwrap_or(0) as u32,
                        ],
                        cx: first_object["bitmap"]["cx"].as_f64().unwrap_or(0.0) as f32,
                        cy: first_object["bitmap"]["cy"].as_f64().unwrap_or(0.0) as f32,
                        w: first_object["bitmap"]["w"].as_f64().unwrap_or(0.0) as f32,
                        h: first_object["bitmap"]["h"].as_f64().unwrap_or(0.0) as f32,
                    },
                },
            };

            // Sauvegarder le JSON nettoyé
            let file_name = path.file_name().unwrap();
            let output_path = output_folder.join(file_name);
            let output_file = File::create(output_path)?;
            serde_json::to_writer_pretty(output_file, &cleaned)?;
        }
    }

    Ok(())
}

fn main() -> std::io::Result<()> {
    let input_root = Path::new("data_clean");
    let output_root = Path::new("data_correct");

    // Créer le dossier output racine
    if !output_root.exists() {
        fs::create_dir(output_root)?;
    }

    // Parcourir les 4 sous-dossiers
    for subfolder in fs::read_dir(input_root)? {
        let entry = subfolder?;
        let input_path = entry.path();

        if input_path.is_dir() {
            let folder_name = input_path.file_name().unwrap();
            let output_path = output_root.join(folder_name);
            println!("📁 Traitement du dossier {:?}", folder_name);

            process_folder(&input_path, &output_path)?;
        }
    }

    println!("✅ Tous les fichiers ont été nettoyés !");
    Ok(())
}
