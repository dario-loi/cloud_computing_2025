import os
import json
import numpy as np
import onnxruntime
from tokenizers import Tokenizer
import traceback

onnx_session = None
tokenizer_obj = None
model_config_data = None
max_seq_length = 128
pad_token_val = 0
id2label_mapping = {}
label_names = []

MODEL_FILES_DIR_NAME = "emo_mobilebert_onnx"


def np_softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def _initialize_resources():
    global onnx_session, tokenizer_obj, model_config_data
    global max_seq_length, pad_token_val, id2label_mapping, label_names

    if onnx_session is not None:
        return

    try:
        print("Initializing resources (cold start or new container)...")
        base_path = "/var/task/"
        model_assets_path = os.path.join(base_path, MODEL_FILES_DIR_NAME)

        config_file = os.path.join(model_assets_path, "config.json")
        with open(config_file, "r", encoding="utf-8") as f:
            model_config_data = json.load(f)

        id2label_mapping = {
            int(k): v for k, v in model_config_data.get("id2label", {}).items()
        }
        if not id2label_mapping:
            raise ValueError("id2label mapping not found or empty in config.json.")

        label_names = [id2label_mapping[i] for i in range(len(id2label_mapping))]
        max_seq_length = model_config_data.get("max_position_embeddings", 128)

        tokenizer_file_path = os.path.join(model_assets_path, "tokenizer.json")
        tokenizer_obj = Tokenizer.from_file(tokenizer_file_path)
        pad_token_val = tokenizer_obj.token_to_id("[PAD]") or 0
        tokenizer_obj.enable_truncation(max_length=max_seq_length)
        tokenizer_obj.enable_padding(
            direction="right",
            length=max_seq_length,
            pad_id=pad_token_val,
            pad_token="[PAD]",
            pad_type_id=0,
        )

        onnx_model_path = os.path.join(model_assets_path, "model.onnx")
        providers = onnxruntime.get_available_providers()
        onnx_session = onnxruntime.InferenceSession(
            onnx_model_path, providers=providers
        )
        print("ONNX session created. Resources initialized.")

    except Exception as e:
        print(f"Initialization error: {e}")
        raise RuntimeError(f"Failed to initialize resources: {str(e)}")


def classify_sentence_with_onnx(sentence_text: str):
    if onnx_session is None:
        _initialize_resources()

    encoded = tokenizer_obj.encode(sentence_text)
    inputs = {
        "input_ids": np.array([encoded.ids], dtype=np.int64),
        "attention_mask": np.array([encoded.attention_mask], dtype=np.int64),
        "token_type_ids": np.array([encoded.type_ids], dtype=np.int64),
    }
    logits = onnx_session.run(None, inputs)[0]
    probs = np_softmax(logits)[0]
    pred_idx = int(np.argmax(probs))
    return {
        "predicted_label": label_names[pred_idx],
        "confidence_score": float(probs[pred_idx]),
        "details": dict(zip(label_names, map(float, probs))),
    }


def lambda_handler(event, context):
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "OPTIONS,POST",
    }

    sentence = None

    if isinstance(event, dict):
        if "body" in event and event["body"]:
            try:
                body_data = json.loads(event["body"])
                sentence = body_data.get("sentence")
            except (json.JSONDecodeError, TypeError):
                sentence = None
        else:
            sentence = event.get("sentence")

    if not sentence or not isinstance(sentence, str) or not sentence.strip():
        print("No valid sentence found, treating as preflight OPTIONS request.")
        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps({"message": "CORS preflight check successful"}),
        }
    try:
        print(f"Processing sentence: {sentence}")
        result = classify_sentence_with_onnx(sentence)
        return {"statusCode": 200, "headers": headers, "body": json.dumps(result)}
    except Exception as e:
        print(f"Classification error: {e}")
        traceback.print_exc()
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps(
                {"error": "Failed to classify sentence", "details": str(e)}
            ),
        }
