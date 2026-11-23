use ct2rs_sys::{ComputeType, Config, Device, StorageView, Whisper, WhisperOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::var("WHISPER_MODEL").unwrap_or_else(|_| "model".to_string());
    println!("Target WHISPER_MODEL: {}", model_path);

    println!("Rust Whisper load: starting Whisper::new");
    let model = Whisper::new(
        &model_path,
        Config {
            compute_type: ComputeType::FLOAT32,
            num_threads_per_replica: 1,
            ..Default::default()
        },
    )?;
    println!("Rust Whisper load: success");

    let mut data = vec![0.0f32; 80 * 16];
    let features = StorageView::new(&[1, 80, 16], &mut data, Device::CPU)?;

    println!("Rust Whisper encode: starting model.encode");
    let encoded = model.encode(&features, true)?;
    println!("Rust Whisper encode: encoded.size() = {}", encoded.size());
    assert_eq!(encoded.size(), 8 * 384, "encoded size mismatch");
    println!("Rust Whisper encode: success");

    println!("Rust Whisper generate: starting model.generate");
    let result = model.generate(
        &encoded,
        &[vec![
            "<|startoftranscript|>",
            "<|en|>",
            "<|transcribe|>",
            "<|notimestamps|>",
        ]],
        &WhisperOptions {
            beam_size: 1,
            max_length: 8,
            ..Default::default()
        },
    )?;
    println!("Rust Whisper generate: result.len() = {}", result.len());
    assert_eq!(result.len(), 1, "result length mismatch");
    assert!(!result[0].sequences_ids.is_empty(), "result sequences_ids is empty");
    println!("Rust Whisper generate: success");

    println!("Rust Whisper PASS");
    Ok(())
}
