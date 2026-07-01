import os
import torch
import tensorrt as trt

_TRT_LOGGER = None

def get_trt_logger():
    global _TRT_LOGGER
    if _TRT_LOGGER is None:
        _TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    return _TRT_LOGGER

def build_trt_engine(onnx_path: str, engine_path: str, fp16: bool = True):
    """
    Builds a TensorRT engine from an ONNX model and saves it.
    """
    logger = get_trt_logger()
    builder = trt.Builder(logger)
    network = builder.create_network()  # explicit batch is default in TRT 10+
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(f"ONNX parse error: {parser.get_error(i)}")
        raise RuntimeError(f"Failed to parse ONNX file: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GiB

    if fp16:
        # TRT 10 flag for FP16 is BuilderFlag.FP16 or PREFER_PRECISION_CONSTRAINTS
        fp16_flag = getattr(trt.BuilderFlag, "FP16",
                    getattr(trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS", None))
        if fp16_flag is not None:
            config.set_flag(fp16_flag)
            print("Set FP16 builder flag.")
        else:
            print("Warning: no FP16 BuilderFlag found in this TRT version")

    print(f"Building serialized network for {onnx_path}...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Failed to build serialized network")

    print(f"Saving TRT engine to {engine_path}...")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print("Engine build and save successful.")

def load_trt_engine(engine_path: str):
    """
    Loads and deserializes a CUDA engine.
    """
    logger = get_trt_logger()
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        serialized = f.read()
    engine = runtime.deserialize_cuda_engine(serialized)
    return engine

def get_torch_dtype(trt_dtype):
    if trt_dtype == trt.DataType.FLOAT:
        return torch.float32
    elif trt_dtype == trt.DataType.HALF:
        return torch.float16
    elif trt_dtype == trt.DataType.INT32:
        return torch.int32
    elif trt_dtype == trt.DataType.INT64:
        return torch.int64
    elif trt_dtype == trt.DataType.BOOL:
        return torch.bool
    elif trt_dtype == trt.DataType.BF16:
        return torch.bfloat16
    elif trt_dtype == trt.DataType.UINT8:
        return torch.uint8
    elif trt_dtype == trt.DataType.INT8:
        return torch.int8
    else:
        raise ValueError(f"Unsupported TRT data type: {trt_dtype}")

class TRTEngineRunner:
    def __init__(self, engine_path: str):
        self.engine = load_trt_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs = {}
        self.outputs = {}
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = get_torch_dtype(dtype)
            
            shape = list(shape)
            
            info = {
                "name": name,
                "shape": shape,
                "dtype": torch_dtype,
            }
            if mode == trt.TensorIOMode.INPUT:
                self.inputs[name] = info
            elif mode == trt.TensorIOMode.OUTPUT:
                self.outputs[name] = info

    def run(self, inputs_dict: dict, output_tensors: dict = None) -> dict:
        """
        Runs async inference on the current PyTorch stream.
        """
        # 1. Bind inputs
        for name, info in self.inputs.items():
            if name not in inputs_dict:
                raise ValueError(f"Missing input tensor: {name}")
            tensor = inputs_dict[name]
            if not tensor.is_cuda:
                raise ValueError(f"Tensor {name} must be on CUDA")
            
            # Set input shape in context in case of dynamic shapes
            self.context.set_input_shape(name, tensor.shape)
            self.context.set_tensor_address(name, tensor.data_ptr())

        # 2. Allocate and bind outputs
        outputs = {}
        for name, info in self.outputs.items():
            if output_tensors and name in output_tensors:
                out_tensor = output_tensors[name]
            else:
                actual_shape = self.context.get_tensor_shape(name)
                out_tensor = torch.empty(tuple(actual_shape), dtype=info["dtype"], device="cuda")
            
            self.context.set_tensor_address(name, out_tensor.data_ptr())
            outputs[name] = out_tensor

        # 3. Execute
        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream)
        return outputs
