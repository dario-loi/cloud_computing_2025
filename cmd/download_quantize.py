from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import (
    AutoQuantizationConfig,
)
import os

model_id = "lordtt13/emo-mobilebert"
task = "sequence-classification"

onnx_output_directory = (
    "emo_mobilebert_onnx_quantized"
)

# Create the output directory if it doesn't exist
if not os.path.exists(onnx_output_directory):
    os.makedirs(onnx_output_directory)

print(f"Starting conversion of {model_id} to ONNX...")

try:

    ort_model = ORTModelForSequenceClassification.from_pretrained(
        model_id, export=True  # Key flag to trigger ONNX conversion
    )
    print("Model loaded and exported to ONNX in memory (non-quantized).")
    ort_model.save_pretrained(onnx_output_directory)
    print(f"Non-quantized ONNX model and config saved to ./{onnx_output_directory}")

    non_quantized_onnx_model_path = os.path.join(onnx_output_directory, "model.onnx")

    if not os.path.exists(non_quantized_onnx_model_path):
        raise FileNotFoundError(
            f"The non-quantized model.onnx was not found at {non_quantized_onnx_model_path} after saving."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(onnx_output_directory)
    print(f"Tokenizer saved to ./{onnx_output_directory}")

    print("\n--- Starting ONNX Model Dynamic Quantization ---")

    quantizer = ORTQuantizer.from_pretrained(ort_model)
    print("ORTQuantizer initialized.")

    dynamic_quantization_config = AutoQuantizationConfig.avx2(
        is_static=False,
        per_channel=False,
    )
    print(f"Dynamic quantization configuration created: {dynamic_quantization_config}")

    quantized_model_path = os.path.join(onnx_output_directory, "model_quantized.onnx")
    print(f"Quantized ONNX model will be saved to: {quantized_model_path}")

    print(f"Applying dynamic quantization to '{non_quantized_onnx_model_path}'...")
    quantizer.quantize(
        save_dir=onnx_output_directory,
        quantization_config=dynamic_quantization_config,
    )
    print(
        f"Dynamic quantization complete. Quantized model saved to: {quantized_model_path}"
    )

    print("\nConversion to ONNX and dynamic quantization finished successfully!")
    print(f"Files are located in: ./{onnx_output_directory}")
    print(f"  - Non-quantized model: model.onnx")
    print(f"  - Quantized model:     model_quantized.onnx")
    print(f"  - Tokenizer and model config files are also in this directory.")

except FileNotFoundError as fnf_error:
    print(f"ERROR - File Not Found: {fnf_error}")
except Exception as e:
    print(f"An error occurred during the process: {e}")
    import traceback

    traceback.print_exc()
