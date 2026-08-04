import os
import sys
import tarfile
import argparse
from datetime import datetime
import shutil

def main():
    parser = argparse.ArgumentParser(description="Package ONNX files and rebuild scripts into a tarball for device deployment.")
    parser.add_argument("--engine-dir", type=str, required=True, help="Directory containing the ONNX and engine files.")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save the generated tarball.")
    parser.add_argument("--fp16", action="store_true", help="Set to rebuild with FP16 precision on device.")
    args = parser.parse_args()

    engine_dir = os.path.abspath(args.engine_dir)
    if not os.path.exists(engine_dir):
        print(f"Error: Engine directory '{engine_dir}' does not exist.")
        sys.exit(1)

    # Find policy name from engine_dir path/contents
    policy_name = os.path.basename(engine_dir)
    date_str = datetime.now().strftime("%Y%m%d")
    tarball_name = f"{policy_name}_engines_{date_str}.tar.gz"
    tarball_path = os.path.join(os.path.abspath(args.output_dir), tarball_name)

    # We will create a temporary packaging directory inside output_dir
    temp_dir = os.path.join(os.path.abspath(args.output_dir), f"_temp_package_{policy_name}")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # 1. Collect all ONNX, ONNX data, and JSON config files from the engine directory
    onnx_files = []
    has_onnx = False
    for filename in os.listdir(engine_dir):
        if filename.endswith(".onnx") or filename.endswith(".onnx.data") or filename.endswith(".json"):
            src_path = os.path.join(engine_dir, filename)
            dst_path = os.path.join(temp_dir, filename)
            shutil.copy2(src_path, dst_path)
            if filename.endswith(".onnx"):
                onnx_files.append(filename)
                has_onnx = True

    if not has_onnx:
        print(f"Error: No ONNX files found in '{engine_dir}'. Make sure to run convert_policy.py first.")
        shutil.rmtree(temp_dir)
        sys.exit(1)

    # 2. Write rebuild_on_device.py
    # - Runs on first load to apply device specific changes to onnx
    py_path = os.path.join(temp_dir, "rebuild_on_device.py")
    with open(py_path, "w") as f:
        f.write('''import os
from lerobot_ros.trt.engine import build_trt_engine

def main():
    fp16 = ''' + str(args.fp16) + '''
    onnx_files = ''' + str(onnx_files) + '''
    for onnx in onnx_files:
        base = os.path.splitext(onnx)[0]
        build_trt_engine(onnx, f"{base}.trt", fp16=fp16)

if __name__ == "__main__":
    main()
''')

    # 3. Generate the config TOML snippet
    #
    # episode_tracker.onnx is the fixed filename convert_policy.py's
    # episode_tracker mode exports (see convert_episode_tracker). Its TRT
    # engine is loaded as a policy's *progress model*, not as the policy
    # itself, so it needs the progress_model_* keys nested inside the
    # policy's own [policies.<name>] block rather than a use_trt/
    # trt_engine_dir snippet of its own.
    is_episode_tracker = "episode_tracker.onnx" in onnx_files
    toml_path = os.path.join(temp_dir, "config_snippet.toml")
    with open(toml_path, "w") as f:
        if is_episode_tracker:
            f.write("# Add the following lines to the [policies.<name>] block of the policy\n")
            f.write("# whose progress model this engine belongs to:\n")
            f.write("progress_model_use_trt = true\n")
            f.write(f"progress_model_trt_engine_dir = \"/path/to/extracted/engines/{policy_name}\"\n")
        else:
            f.write("# Add the following lines to your policy configuration TOML file:\n")
            f.write(f"[policies.{policy_name}]\n")
            f.write("pretrained_name_or_path = \"/path/to/original_pytorch_checkpoint\"\n")
            f.write("device = \"cuda\"\n")
            f.write("use_trt = true\n")
            f.write(f"trt_engine_dir = \"/path/to/extracted/engines/{policy_name}\"\n")
            f.write(f"trt_fp16 = {str(args.fp16).lower()}\n")

    # Create the tarball
    print(f"Packaging files into {tarball_path}...")
    with tarfile.open(tarball_path, "w:gz") as tar:
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            tar.add(file_path, arcname=os.path.join(policy_name, filename))

    # Clean up temp dir
    shutil.rmtree(temp_dir)
    print("Packaging completed successfully!")

if __name__ == "__main__":
    main()
