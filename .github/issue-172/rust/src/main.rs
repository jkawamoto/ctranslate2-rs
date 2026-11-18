use ct2rs_sys::{ComputeType, Config, Device, StorageView, Whisper, WhisperOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Rust Whisper load");
    let model = Whisper::new(
        std::env::var("WHISPER_MODEL")?,
        Config {
            compute_type: ComputeType::FLOAT32,
            num_threads_per_replica: 1,
            ..Default::default()
        },
    )?;
    let mut data = vec![0.; 80 * 16];
    let features = StorageView::new(&[1, 80, 16], &mut data, Device::CPU)?;
    println!("Rust Whisper encode");
    let encoded = model.encode(&features, true)?;
    assert_eq!(encoded.size(), 8 * 384);
    println!("Rust Whisper generate");
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
    assert_eq!(result.len(), 1);
    assert!(!result[0].sequences_ids.is_empty());
    println!("Rust Whisper PASS");
    Ok(())
}
