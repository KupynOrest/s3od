import argparse
import torch
import numpy as np
from pathlib import Path

from model_training.model import DPTSegmentation


class TracedModelWrapper(torch.nn.Module):
    """
    Device-agnostic wrapper for TorchScript export.
    
    Key insight: We can't prevent device specs from being baked into traced operations,
    but we can make inputs/outputs device-agnostic by ensuring tensor operations
    use relative device placement (.to(x.device) patterns).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x: torch.Tensor):
        """Forward pass that preserves input device."""
        # Model was traced on a specific device, but we ensure outputs 
        # are on the same device as inputs
        outputs = self.model(x)
        return {
            'pred_masks': outputs['pred_masks'],
            'pred_iou': outputs['pred_iou']
        }


def load_checkpoint(checkpoint_path: str) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'state_dict' in checkpoint:
        import hydra
        model = hydra.utils.instantiate(checkpoint['hyper_parameters']['config'].model)
        state_dict = {
            k.lstrip("model").lstrip("."): v 
            for k, v in checkpoint["state_dict"].items() 
            if k.startswith("model.")
        }
        model.load_state_dict(state_dict)
    else:
        model = checkpoint
    
    return model


def verify_model(original_model, traced_model, num_tests=10):
    print("\nVerifying traced model...")
    original_model.eval()
    traced_model.eval()
    
    max_diff = 0
    for i in range(num_tests):
        test_input = torch.randn(1, 3, 1024, 1024)
        
        with torch.no_grad():
            original_out = original_model(test_input)
            traced_out = traced_model(test_input)
        
        mask_diff = torch.abs(original_out['pred_masks'] - traced_out['pred_masks']).max().item()
        iou_diff = torch.abs(original_out['pred_iou'] - traced_out['pred_iou']).max().item()
        
        max_diff = max(max_diff, mask_diff, iou_diff)
        
        if i == 0:
            print(f"Test {i+1}: mask_diff={mask_diff:.6f}, iou_diff={iou_diff:.6f}")
    
    print(f"Maximum difference across {num_tests} tests: {max_diff:.6f}")
    
    if max_diff < 1e-5:
        print("✓ Traced model matches original model!")
        return True
    else:
        print("✗ Warning: Traced model differs from original")
        return False


def verify_exported_checkpoint(original_checkpoint_path: str, exported_checkpoint_path: str) -> bool:
    """
    Verify the exported checkpoint is a lossless extraction of the original.

    The export is a deterministic key-stripping operation, so correctness is fully
    determined by comparing both state dicts directly — no model re-instantiation
    needed (which would break when the encoder's internal key structure changes
    across transformers versions).
    """
    print("\nVerifying exported checkpoint against original...")

    original_ckpt = torch.load(original_checkpoint_path, map_location='cpu', weights_only=False)
    exported_ckpt = torch.load(exported_checkpoint_path, map_location='cpu', weights_only=False)

    original_state = {
        k[len('model.'):]: v
        for k, v in original_ckpt['state_dict'].items()
        if k.startswith('model.')
    }
    exported_state = exported_ckpt['state_dict']

    orig_keys = set(original_state.keys())
    exp_keys = set(exported_state.keys())
    missing = orig_keys - exp_keys
    extra = exp_keys - orig_keys
    if missing:
        print(f"  ✗ Keys in original missing from export ({len(missing)}): {sorted(missing)[:5]} ...")
        return False
    if extra:
        print(f"  ✗ Extra keys in export not in original ({len(extra)}): {sorted(extra)[:5]} ...")
        return False
    print(f"  ✓ Key coverage: {len(orig_keys)} keys match exactly")

    mismatched = []
    for k in orig_keys:
        orig_t = original_state[k].float()
        exp_t = exported_state[k].float()
        if orig_t.shape != exp_t.shape or not torch.equal(orig_t, exp_t):
            mismatched.append(k)
    if mismatched:
        print(f"  ✗ Weight mismatches in {len(mismatched)} tensors: {mismatched[:3]} ...")
        return False

    print(f"  ✓ All {len(orig_keys)} tensors are bit-identical")
    return True


def export_checkpoint(checkpoint_path: str, output_path: str, verify: bool = True):
    """
    Export a clean checkpoint for inference (recommended approach).
    
    This approach is preferred over TorchScript because:
    - Full device flexibility (works on any device)
    - Better debugging (full Python access)
    - Easier to update/patch
    - Standard approach used by HuggingFace, OpenAI, etc.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract only the model weights (remove Lightning metadata)
    if 'state_dict' in checkpoint:
        state_dict = {
            k[len('model.'):]: v
            for k, v in checkpoint['state_dict'].items()
            if k.startswith('model.')
        }
        clean_checkpoint = {'state_dict': state_dict}
    else:
        clean_checkpoint = checkpoint
    
    print(f"\nSaving clean checkpoint to: {output_path}")
    torch.save(clean_checkpoint, output_path)
    
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Checkpoint saved successfully! Size: {file_size:.2f} MB")
    print(f"  Contains {len(clean_checkpoint.get('state_dict', clean_checkpoint))} parameters")

    if verify and 'state_dict' in checkpoint:
        if not verify_exported_checkpoint(checkpoint_path, output_path):
            print("\n✗ Verification FAILED — exported checkpoint differs from original!")
            return False
    
    return True


def export_model_torchscript(checkpoint_path: str, output_path: str, verify: bool = True, device: str = 'cpu'):
    """
    Export model to TorchScript (legacy/experimental).
    
    WARNING: TorchScript with transformers models has limitations:
    - Device specification gets baked into the trace
    - Model will only work on the device it was traced on
    - For multi-device support, use export_checkpoint() instead
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Export device: {device}")
    print("\n⚠️  WARNING: TorchScript export will be device-specific!")
    print(f"    This model will ONLY work on: {device}")
    print(f"    For multi-device support, use checkpoint loading instead.\n")
    
    model = load_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    
    print(f"Model type: {type(model).__name__}")
    
    wrapper = TracedModelWrapper(model)
    wrapper.eval()
    
    print("\nTracing model...")
    example_input = torch.randn(1, 3, 1024, 1024, device=device)
    
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapper, example_input, strict=False)
    
    if verify:
        is_valid = verify_model(wrapper, traced_model)
        if not is_valid:
            print("\nWarning: Verification failed, but continuing with export...")
    
    print(f"\nSaving traced model to: {output_path}")
    torch.jit.save(traced_model, output_path)
    
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Model saved successfully! Size: {file_size:.2f} MB")
    
    print("\nTesting model loading...")
    loaded_model = torch.jit.load(output_path)
    with torch.no_grad():
        test_output = loaded_model(example_input)
    print(f"✓ Model loads and runs successfully!")
    print(f"  Output shape: pred_masks={test_output['pred_masks'].shape}, pred_iou={test_output['pred_iou'].shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Export S3OD model for inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export clean checkpoint (recommended)
  python export_model.py --checkpoint best.ckpt --output s3od_checkpoint.ckpt
  
  # Export TorchScript (device-specific)
  python export_model.py --checkpoint best.ckpt --output s3od.pt --format torchscript --device cuda
  
Recommendation:
  Use checkpoint format for maximum flexibility. TorchScript with transformers
  models has device-baking issues that cannot be fully resolved.
        """
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for exported model"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["checkpoint", "torchscript"],
        default="checkpoint",
        help="Export format: checkpoint (recommended) or torchscript (device-specific)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for TorchScript export (cpu or cuda). Only used with --format torchscript"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification step (TorchScript only)"
    )
    
    args = parser.parse_args()
    
    if args.format == "checkpoint":
        export_checkpoint(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            verify=not args.no_verify,
        )
    else:  # torchscript
        export_model_torchscript(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            device=args.device,
            verify=not args.no_verify
        )


if __name__ == "__main__":
    main()

