import time
import numpy as np
from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig
import openvino.runtime as ov


import warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="optimum.intel.openvino.modeling_decoder"
)

ov_config = {
    "INFERENCE_NUM_THREADS": 1,   # Use 8 CPU threads
    # "ENABLE_CPU_PINNING": "YES",
    # "CPU_DENORMALS_OPTIMIZATION":"YES"      # Enable CPU pinning for better performance
    # "PERFORMANCE_HINT": "LATENCY",
    # "ENABLE_HYPER_THREADING": "NO", 
    # "NUM_STREAMS" : "2" # Optimize for lowest latency
    "PERFORMANCE_HINT":"THROUGHPUT"
}



DEVICE = "CPU"

# Try to import pyRAPL for energy measurements; if not installed, instruct to install it.
try:
    import pyRAPL
    pyRAPL.setup()
    ENERGY_MEASUREMENT = True
    m = pyRAPL.Measurement("Inference")
except ImportError:
    print("pyRAPL not installed. Install it via 'pip install pyRAPL' to measure energy consumption.")
    ENERGY_MEASUREMENT = False

# --- Helper function to run inference and measure performance ---
def run_inference_experiment(model, tokenizer, inputs, max_new_tokens=50):
    """
    Runs inference on a list of tokenized inputs.
    Measures total inference time and total energy (if pyRAPL available).
    Returns average latency per sample, throughput (samples/second),
    and average EDP (energy in Joules * latency in seconds per sample).
    """
    total_time = 0.0
    total_energy = 0.0
    num_samples = len(inputs)
    
    # Warm-up: run a few inferences (not measured) so that caching effects settle.
    for _ in range(3):
        _ = model.generate(**inputs[0], max_new_tokens=max_new_tokens)
    
    for inp in inputs:
        # Measure energy and time for each inference:
        if ENERGY_MEASUREMENT:
            start_time = time.time()
            #with pyRAPL.Measurement("inference") as m:
            m.begin()
            _ = model.generate(**inp, max_new_tokens=max_new_tokens)
            elapsed = time.time() - start_time
            m.end()
            if m is not None:
                print("Getting energy computed from RAPL")
                energy = m.result.pkg[0] / 1e6 # energy in Joules (if available)
            else:
                print("Using default value for energy")
                energy = 120
        else:
            start_time = time.time()
            _ = model.generate(**inp, max_new_tokens=max_new_tokens)
            elapsed = time.time() - start_time
            energy = 0.0  # fallback if not measuring energy
        total_time += elapsed
        total_energy += energy

    avg_latency = total_time / num_samples
    throughput = num_samples / total_time
    # Energy-Delay Product per sample (Joule-second); if energy not measured, EDP=0.
    edp = (total_energy / num_samples) * avg_latency

    return {
        "total_time": total_time,
        "avg_latency": avg_latency,
        "throughput": throughput,
        "total_energy": total_energy,
        "edp": edp
    }

# --- Prepare sample input data ---
def prepare_sample_inputs(tokenizer, sample_text, num_samples=20, max_length=64):
    """
    Tokenizes the sample_text (or different texts) to generate a list of inputs.
    Each input will have at most max_length tokens.
    """
    inputs = []
    for _ in range(num_samples):
        tokenized = tokenizer(sample_text,
                                max_length=max_length,
                                truncation=True,
                                padding="do_not_pad",   # For batch size = 1, no need to pad
                                return_tensors="pt")
        inputs.append(tokenized)
    return inputs

# --- Experiment Parameters ---
#MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"  # Example model (from Hugging Face)
MODEL_ID = "gpt2"
SAMPLE_TEXT = ("In today's rapidly evolving technological landscape, "
               "machine learning and artificial intelligence play a pivotal role "
               "in shaping innovative solutions and improving efficiency.")  # Example text

NUM_SAMPLES = 20
MAX_TOKEN_LENGTH = 64
MAX_NEW_TOKENS = 50

# --- Baseline Experiment: FP32 model on CPU ---
print("\n=== Baseline Experiment: FP32 model on CPU ===")
# Load model and tokenizer from a previously converted (or on-the-fly exported) OpenVINO IR.
# Here we assume you have already exported and saved the model. If not, you can export as shown below.
model_fp32 = OVModelForCausalLM.from_pretrained(MODEL_ID, export=True, device=DEVICE,ov_config=ov_config)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# Optionally save for future faster loading:
#model_fp32.save_pretrained("ov_llama_fp32")
#tokenizer.save_pretrained("ov_llama_fp32")

inputs = prepare_sample_inputs(tokenizer, SAMPLE_TEXT, num_samples=NUM_SAMPLES, max_length=MAX_TOKEN_LENGTH)

baseline_metrics = run_inference_experiment(model_fp32, tokenizer, inputs, max_new_tokens=MAX_NEW_TOKENS)

print("Baseline Metrics:")
print(f"  Total Inference Time: {baseline_metrics['total_time']:.3f} s")
print(f"  Average Latency per Sample: {baseline_metrics['avg_latency']*1000:.1f} ms")
print(f"  Throughput: {baseline_metrics['throughput']:.2f} samples/s")
print(f"  Total Energy: {baseline_metrics['total_energy']:.3f} Joules")
print(f"  EDP per Sample: {baseline_metrics['edp']:.6f} Joule-s")

# --- Optimized Experiment: Quantized model with GPU offloading ---
print("\n=== Optimized Experiment: Quantized model on CPU ===")
# Set up a quantization configuration; here we use INT8 (8 bits) quantization.
quant_config = OVWeightQuantizationConfig(bits=8)
# Load and convert the model with quantization applied.
model_quant = OVModelForCausalLM.from_pretrained(MODEL_ID, export=True,
                                                 quantization_config=quant_config,
                                                 device=DEVICE)

# # Save the quantized model for future use.
# #model_quant.save_pretrained("ov_llama_int8")
# # You can reuse the same tokenizer.
optimized_metrics = run_inference_experiment(model_quant, tokenizer, inputs, max_new_tokens=MAX_NEW_TOKENS)

# #---- Optimized Experiment 2: CPU optimization

# # First, define CPU runtime properties to utilize all 16 cores:
# cpu_properties = {
#     "CPU_THROUGHPUT_STREAMS": "16",  # Number of parallel streams
#     "CPU_THREADS_NUM": "16",         # Use all 16 threads/cores
#     "CPU_BIND_THREAD": "YES"         # Enable thread affinity
# }

# # Update runtime configuration with our CPU properties.
# # (This API may vary between releases. Here we assume runtime_config.update exists.)
# try:
#     model_quant.runtime_config.update(cpu_properties)
#     print("Updated model runtime configuration with CPU properties.")
# except Exception as e:
#     print("Could not update runtime configuration directly. Proceeding with default settings.")
#     print(e)


# optimized_metrics_1 = run_inference_experiment(model_quant, tokenizer, inputs, max_new_tokens=MAX_NEW_TOKENS)


# print("Optimized Metrics_1: Quantization")
# print(f"  Total Inference Time: {optimized_metrics_1['total_time']:.3f} s")
# print(f"  Average Latency per Sample: {optimized_metrics_1['avg_latency']*1000:.1f} ms")
# print(f"  Throughput: {optimized_metrics_1['throughput']:.2f} samples/s")
# print(f"  Total Energy: {optimized_metrics_1['total_energy']:.3f} Joules")
# print(f"  EDP per Sample: {optimized_metrics_1['edp']:.6f} Joule-s")


print("Optimized Metrics: Quantization")
print(f"  Total Inference Time: {optimized_metrics['total_time']:.3f} s")
print(f"  Average Latency per Sample: {optimized_metrics['avg_latency']*1000:.1f} ms")
print(f"  Throughput: {optimized_metrics['throughput']:.2f} samples/s")
print(f"  Total Energy: {optimized_metrics['total_energy']:.3f} Joules")
print(f"  EDP per Sample: {optimized_metrics['edp']:.6f} Joule-s")

# --- Comparison Summary ---
print("\n=== Comparison Summary ===")
print(f"Baseline (CPU FP32) Avg Latency: {baseline_metrics['avg_latency']*1000:.1f} ms, "
      f"Throughput: {baseline_metrics['throughput']:.2f} samples/s, EDP: {baseline_metrics['edp']:.6f} Js")
print(f"Optimized (CPU INT8) Avg Latency: {optimized_metrics['avg_latency']*1000:.1f} ms, "
      f"Throughput: {optimized_metrics['throughput']:.2f} samples/s, EDP: {optimized_metrics['edp']:.6f} Js")
