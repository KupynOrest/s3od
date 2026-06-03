import time
from pathlib import Path
import numpy as np
import torch
import torch.profiler
from thop import profile, clever_format
from tqdm import tqdm
import fire
import copy

from model_training.predictor import SODPredictor


class BenchmarkSOD:
    """Benchmark SODPredictor performance metrics."""

    def __init__(self):
        self.SODPredictor = SODPredictor

    def count_parameters(self, model):
        """Count number of trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def measure_fps(self, predictor, input_size=840, num_iterations=100):
        """
        Measure FPS by running multiple forward passes of just the model

        Args:
            predictor: SODPredictor instance
            input_size: Tuple of (height, width) for input image
            num_iterations: Number of iterations for measurement
        """
        # Create dummy tensor that matches the expected input shape after preprocessing
        dummy_input = torch.randn(1, 3, input_size, input_size).to(predictor.device)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                predictor.model(dummy_input)

        # Measure time
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            for _ in tqdm(range(num_iterations), desc="Measuring FPS"):
                predictor.model(dummy_input)
                torch.cuda.synchronize()

        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = num_iterations / elapsed_time

        return fps

    def measure_flops(self, predictor):
        """
        Measure FLOPs using thop

        Args:
            predictor: SODPredictor instance
        """
        # Create a copy of the model for FLOPs counting to avoid hook conflicts
        model_copy = copy.deepcopy(predictor.model)
        model_copy.to(predictor.device)

        dummy_input = torch.randn(1, 3, 840, 840).to(predictor.device)

        try:
            flops, params = profile(model_copy, inputs=(dummy_input,), verbose=False)
            flops, params = clever_format([flops, params], "%.3f")
        finally:
            # Clean up the model copy
            del model_copy
            torch.cuda.empty_cache()

        return flops, params

    def measure_memory_usage(self, predictor, input_size=(840, 840)):
        """
        Profile memory usage using torch.profiler

        Args:
            predictor: SODPredictor instance
            input_size: Tuple of (height, width) for input image
        """
        dummy_input = np.random.randint(0, 255, (*input_size, 3), dtype=np.uint8)

        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                profile_memory=True,
                record_shapes=True,
                with_stack=True
        ) as prof:
            result = predictor.predict(dummy_input)
            torch.cuda.synchronize()  # Ensure all CUDA operations are completed

        return prof

    def get_memory_stats(self):
        """Get current GPU memory statistics"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
            max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            return {
                "current_allocated": f"{memory_allocated:.2f} MB",
                "max_allocated": f"{max_memory_allocated:.2f} MB",
                "reserved": f"{memory_reserved:.2f} MB"
            }
        return {"error": "CUDA not available"}

    def run(
            self,
            checkpoint: str,
            image_size: int = 840,
            iterations: int = 100,
            output: str = "benchmark_results.txt",
    ):
        """
        Run all benchmarks and save results

        Args:
            checkpoint: Path to model checkpoint
            image_size: Input image size (both height and width)
            iterations: Number of iterations for FPS measurement
            output: Output file path for results
        """
        print("\n=== Starting Benchmark ===")

        # Initialize predictor
        predictor = self.SODPredictor(
            checkpoint_path=checkpoint,
            image_size=image_size
        )

        try:
            # Measure FPS
            print("\nMeasuring FPS...")
            fps = self.measure_fps(predictor, input_size=image_size, num_iterations=iterations)

            # Count parameters
            num_params = self.count_parameters(predictor.model)

            # Measure FLOPs
            print("\nMeasuring FLOPs...")
            flops, params = self.measure_flops(predictor)

            # Get memory stats before profiling
            print("\nGetting memory statistics...")
            memory_stats = self.get_memory_stats()

            # Profile memory
            print("\nProfiling memory usage...")
            prof = self.measure_memory_usage(predictor)

            # Save results
            output_path = Path(output)
            with open(output_path, 'w') as f:
                f.write("=== SODPredictor Benchmark Results ===\n\n")
                f.write(f"Model Parameters: {num_params:,}\n")
                f.write(f"FLOPs: {flops}\n")
                f.write(f"FPS: {fps:.2f}\n")
                f.write(f"Inference time per image: {1000 / fps:.2f}ms\n\n")

                f.write("=== Memory Statistics ===\n")
                for key, value in memory_stats.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

                f.write("=== Memory Profile ===\n")
                f.write(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

                # Add stack traces for top memory-consuming operations
                f.write("\n=== Top Memory-Consuming Operations (with stack traces) ===\n")
                f.write(prof.key_averages(group_by_stack_n=5).table(
                    sort_by="self_cuda_memory_usage",
                    row_limit=5
                ))

            print(f"\nResults saved to {output_path}")

            # Print summary to console
            print("\n=== Summary ===")
            print(f"FPS: {fps:.2f}")
            print(f"Parameters: {num_params:,}")
            print(f"FLOPs: {flops}")
            for key, value in memory_stats.items():
                print(f"Memory {key}: {value}")
            print(f"Detailed results saved to: {output_path}")

        finally:
            # Cleanup
            torch.cuda.empty_cache()


def compute_metrics(checkpoint="model.pth", image_size=840, iterations=100, output="benchmark_results.txt"):
    benchmark = BenchmarkSOD()
    benchmark.run(checkpoint, image_size, iterations, output)


if __name__ == "__main__":
    fire.Fire(compute_metrics)
