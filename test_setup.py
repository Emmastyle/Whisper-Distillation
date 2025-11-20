"""Test script to verify the setup is working"""
import torch
from src.data import WhisperASRDataset, whisper_collate_fn
from src.models import TeacherWhisper, StudentASRModel
from src.utils.config import load_config
from torch.utils.data import DataLoader

def test_setup():
    """Test basic setup"""
    print("=" * 60)
    print("Testing Phase 1 Setup")
    print("=" * 60)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Device: {device}")
    
    # Test config loading
    print("\n[1/4] Testing config loading...")
    config = load_config("configs/ce_tiny_s2.yaml")
    print(f"✓ Config loaded successfully")
    print(f"  - Batch size: {config.data.batch_size}")
    print(f"  - Student decoder layers: {config.model.student_decoder_layers}")
    print(f"  - Learning rate: {config.training.learning_rate}")
    
    # Test dataset
    print("\n[2/4] Testing dataset loading (loading 10 samples)...")
    try:
        dataset = WhisperASRDataset(
            dataset_name=config.data.dataset_name,
            dataset_config=config.data.dataset_config,
            split=config.data.split,
            max_duration=config.data.max_duration,
            num_samples=10,  # Test with just 10 samples
            sample_rate=config.data.sample_rate,
        )
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Test single sample
        sample = dataset[0]
        print(f"  - Input features shape: {sample['input_features'].shape}")
        print(f"  - Labels shape: {sample['labels'].shape}")
        print(f"  - Text: {sample['text'][:50]}...")
        
        # Test dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=whisper_collate_fn,
        )
        batch = next(iter(dataloader))
        print(f"✓ DataLoader working")
        print(f"  - Batch input features shape: {batch['input_features'].shape}")
        print(f"  - Batch labels shape: {batch['labels'].shape}")
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False
    
    # Test models
    print("\n[3/4] Testing model loading...")
    try:
        teacher = TeacherWhisper(config.model.teacher_model)
        print(f"✓ Teacher model loaded")
        print(f"  - Vocab size: {teacher.vocab_size}")
        print(f"  - Hidden size: {teacher.d_model}")
        
        model = StudentASRModel(
            teacher=teacher,
            student_decoder_layers=config.model.student_decoder_layers,
            student_decoder_heads=config.model.student_decoder_heads,
            dropout=config.model.dropout,
        )
        print(f"✓ Student model created")
        
        # Count parameters
        student_params = sum(p.numel() for p in model.student_decoder.parameters())
        trainable_params = sum(p.numel() for p in model.student_decoder.parameters() if p.requires_grad)
        print(f"  - Student decoder parameters: {student_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False
    
    # Test forward pass
    print("\n[4/4] Testing forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            input_features = batch['input_features']
            labels = batch['labels']
            
            # Prepare decoder input
            from src.training.train_ce import prepare_decoder_input_ids
            from transformers import WhisperTokenizer
            tokenizer = WhisperTokenizer.from_pretrained(config.model.teacher_model)
            decoder_input_ids = prepare_decoder_input_ids(labels, tokenizer.bos_token_id)
            
            # Forward
            loss, logits = model(
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
            )
            
            print(f"✓ Forward pass successful")
            print(f"  - Loss: {loss.item():.4f}")
            print(f"  - Logits shape: {logits.shape}")
            
            # Test generation
            generated = model.generate(
                input_features=input_features[:1],
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            print(f"✓ Generation successful")
            print(f"  - Generated shape: {generated.shape}")
            
            # Decode
            text = tokenizer.batch_decode(generated, skip_special_tokens=True)
            print(f"  - Generated text: {text[0][:50]}...")
            
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nYou can now run training with:")
    print("  python -m src.training.train_ce --config configs/ce_tiny_s2.yaml")
    print("  python -m src.training.train_kd --config configs/kd_tiny_s2.yaml")
    print("\nOr use the scripts:")
    print("  bash scripts/train_ce.sh")
    print("  bash scripts/train_kd.sh")
    return True

if __name__ == "__main__":
    test_setup()

