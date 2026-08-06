import os
import torch
import tensorrt as trt

_TRT_LOGGER = None

def get_trt_logger():
    global _TRT_LOGGER
    if _TRT_LOGGER is None:
        _TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    return _TRT_LOGGER

def convert_onnx_to_fp16(src_path: str, dst_path: str):
    """
    Convert an FP32 ONNX model to a fully-FP16, TensorRT-11-parseable ONNX.

    Why not ``keep_io_types=True``: onnxconverter_common leaves graph I/O (and a few
    dynamic-shape Cast nodes) in FP32, producing Float/Half edges. TensorRT 11's ONNX
    parser strictly validates elementwise/matmul input types (no implicit promotion),
    so it rejects those edges (``ElementWiseOperation DIV must have same input types``).

    Fix: convert *everything* to FP16 (``keep_io_types=False``) and then flatten the
    residual ``Cast(to=FLOAT)`` nodes — emitted for dynamic-dimension math like
    ``1.0 / num_patches`` — to ``Cast(to=FLOAT16)``. The result is type-consistent and
    builds as a STRONGLY_TYPED engine. The engine I/O is therefore FP16;
    ``TRTEngineRunner`` casts host-provided tensors to the expected dtype so callers
    still hand it FP32 noise/state/timestep unchanged.

    Two subtleties, both of which TensorRT rejects the graph over if skipped:

    * Shape inference has to run first. ``convert_float_to_float16`` can only place
      the fp32/fp16 boundary casts where it knows tensor types; with inference off it
      leaves the ops in ``DEFAULT_OP_BLOCK_LIST`` (Resize, Range, CumSum, ...) with
      FP32 outputs, which then propagate through type-agnostic ops (Pad, Unsqueeze)
      into FP16 arithmetic — ``ElementWiseOperation PROD must have same input types.
      But they are of types Float and Half``.
    * The flattening must not touch casts feeding those blocked ops. They are there
      on purpose and TensorRT agrees with them: its Range importer takes only
      int32/int64/fp32 and rejects FP16 with ``UNSUPPORTED_NODE: (isInt32 ||
      isInt64 || isFp32)``.

    Handles models with external weight data (the SmolVLM backbone is > 1 GB), which
    is why shape inference goes through the file-based ``infer_shapes_path``: the
    in-memory entry point cannot carry a model past the 2 GB protobuf limit.
    """
    import onnx
    from onnxconverter_common import float16

    # Written beside the source so the external-data references, which are
    # relative to the model file, still resolve from the inferred copy.
    inferred_path = f"{os.path.splitext(src_path)[0]}_inferred.onnx"
    onnx.shape_inference.infer_shapes_path(src_path, inferred_path, strict_mode=False)
    try:
        model = onnx.load(inferred_path, load_external_data=True)
    finally:
        if os.path.exists(inferred_path):
            os.remove(inferred_path)

    fp16_model = float16.convert_float_to_float16(
        model,
        keep_io_types=False,
        disable_shape_infer=False,
    )

    blocked_ops = set(float16.DEFAULT_OP_BLOCK_LIST)
    consumers = {}
    for node in fp16_model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    rewritten = kept = 0
    for node in fp16_model.graph.node:
        if node.op_type != "Cast":
            continue
        if any(c.op_type in blocked_ops for c in consumers.get(node.output[0], [])):
            kept += 1
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i == int(onnx.TensorProto.FLOAT):
                attr.i = int(onnx.TensorProto.FLOAT16)
                rewritten += 1
    print(
        f"Rewrote {rewritten} Cast(FLOAT->FLOAT16) for TRT type consistency; "
        f"kept {kept} feeding FP32-only ops"
    )
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    onnx.save(
        fp16_model,
        dst_path,
        save_as_external_data=True,
        location=os.path.basename(dst_path) + ".data",
        all_tensors_to_one_file=True,
    )
    print(f"Converted FP16 ONNX saved to {dst_path}")


def build_trt_engine(onnx_path: str, engine_path: str, fp16: bool = True):
    """
    Builds a TensorRT engine from an ONNX model and saves it.

    FP16 handling is version-aware:
      * TRT <= 10 exposed ``BuilderFlag.FP16`` (weak typing: FP32 network, builder
        auto-picks reduced precision). We set that flag directly.
      * TRT 11 removed all precision builder flags. Precision comes from the *network
        types*, so for FP16 we convert the ONNX to FP16 (``convert_onnx_to_fp16``) and
        build a STRONGLY_TYPED network from it. The engine I/O becomes FP16, which
        ``TRTEngineRunner`` handles by casting inputs to the expected dtype.

    Either way the call site is unchanged: ``build_trt_engine(onnx, trt, fp16=True)``.
    """
    logger = get_trt_logger()

    has_fp16_flag = hasattr(trt.BuilderFlag, "FP16")
    strongly_typed = fp16 and not has_fp16_flag

    if strongly_typed:
        # TRT 11+: bake FP16 into the ONNX and build a strongly-typed network.
        root, ext = os.path.splitext(onnx_path)
        fp16_onnx = f"{root}_fp16{ext}"
        convert_onnx_to_fp16(onnx_path, fp16_onnx)
        onnx_path = fp16_onnx

    builder = trt.Builder(logger)
    net_flags = 0
    if strongly_typed:
        net_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(net_flags)  # explicit batch is default in TRT 10+
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(f"ONNX parse error: {parser.get_error(i)}")
        raise RuntimeError(f"Failed to parse ONNX file: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GiB

    if strongly_typed:
        print("Building STRONGLY_TYPED FP16 network (precision follows ONNX types).")
    elif fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Set FP16 builder flag.")

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
        # Keep casted/contiguous input tensors alive until the next run() so their
        # GPU memory is not freed before the (async) engine has consumed it. Stream
        # ordering guarantees correctness across successive run() calls.
        self._keepalive = []

        # 1. Bind inputs
        for name, info in self.inputs.items():
            if name not in inputs_dict:
                raise ValueError(f"Missing input tensor: {name}")
            tensor = inputs_dict[name]
            if not tensor.is_cuda:
                raise ValueError(f"Tensor {name} must be on CUDA")

            # Cast to the dtype the engine expects (FP16 engines take FP16 I/O) so
            # callers can keep passing FP32 noise/state/timestep unchanged.
            if tensor.dtype != info["dtype"]:
                tensor = tensor.to(info["dtype"])
            tensor = tensor.contiguous()
            self._keepalive.append(tensor)

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
