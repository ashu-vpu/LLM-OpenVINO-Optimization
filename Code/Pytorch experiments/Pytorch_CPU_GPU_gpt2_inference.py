import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyRAPL
# Check for CUDA availability
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
# Try to import pyRAPL for energy measurements; if not installed, instruct to install it.


pyRAPL.setup()
m = pyRAPL.Measurement("Inference")




def run_inference_experiment(model, tokenizer, inputs, max_new_tokens=50):
    """
    Runs inference on a list of tokenized inputs using PyTorch.
    Measures total inference time and calculates average latency per sample and throughput.
    """
    model.to(DEVICE).eval()  # Move model to correct device and set to eval mode
    total_time = 0.0
    total_energy = 0.0
    num_samples = len(inputs)
    
    # Warm-up: run a few inferences (not measured) to stabilize performance
    with torch.no_grad():
        for _ in range(3):
            _ = model.generate(**{k: v.to(DEVICE) for k, v in inputs[0].items()}, max_new_tokens=max_new_tokens)
            # _ = model.generate(**inp, max_new_tokens=max_new_tokens)
    
    # Measure inference time
    with torch.no_grad():
        for inp in inputs:
            start_time = time.time()
            m.begin()
            _ = model.generate(**{k: v.to(DEVICE) for k, v in inp.items()}, max_new_tokens=max_new_tokens)
            # _ = model.generate(**inp, max_new_tokens=max_new_tokens)
            elapsed = time.time() - start_time
            m.end()
            total_time += elapsed
            energy = m.result.pkg[0] / 1e6 # energy in Joules (if available)
            total_energy += energy
    
    avg_latency = total_time / num_samples
    throughput = num_samples / total_time
    edp = (total_energy / num_samples) * avg_latency
    
    return {
        "total_time": total_time,
        "avg_latency": avg_latency,
        "throughput": throughput,
        "total_energy": total_energy,
        "edp": edp
    }

# Prepare sample inputs
def prepare_sample_inputs(tokenizer, sample_text, num_samples=20, max_length=64):
    """
    Tokenizes the sample_text and returns a list of tokenized inputs.
    """
    inputs = []
    for _ in range(num_samples):
        tokenized = tokenizer(sample_text,
                              max_length=max_length,
                              truncation=True,
                              padding="do_not_pad",  # No padding needed for batch size = 1
                              return_tensors="pt")
        inputs.append(tokenized)
    return inputs

# Experiment Parameters
MODEL_ID = "gpt2"  # You can replace this with a different Hugging Face model
SAMPLE_TEXT = ("In today's rapidly evolving technological landscape, "
               "machine learning and artificial intelligence play a pivotal role "
               "in shaping innovative solutions and improving efficiency.")
NUM_SAMPLES = 20
MAX_TOKEN_LENGTH = 64
MAX_NEW_TOKENS = 50

# Load model and tokenizer
print("\n=== Running Experiment: PyTorch Model on GPU/CPU ===")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Prepare inputs
inputs = prepare_sample_inputs(tokenizer, SAMPLE_TEXT, num_samples=NUM_SAMPLES, max_length=MAX_TOKEN_LENGTH)

# Run inference experiment
metrics = run_inference_experiment(model, tokenizer, inputs, max_new_tokens=MAX_NEW_TOKENS)

# Print results
print("Experiment Metrics:")
print(f"  Total Inference Time: {metrics['total_time']:.3f} s")
print(f"  Average Latency per Sample: {metrics['avg_latency'] * 1000:.1f} ms")
print(f"  Throughput: {metrics['throughput']:.2f} samples/s")
print(f"  Total Energy: {metrics['total_energy']:.3f} Joules")
print(f"  EDP per Sample: {metrics['edp']:.6f} Joule-s")

