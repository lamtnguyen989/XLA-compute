bazel build -c opt --config=cuda \
  --repo_env TF_NEED_CUDA=1 \
  --repo_env TF_NEED_ROCM=0 \
  --action_env CUDA_TOOLKIT_PATH="/usr/local/cuda-13.3" \
  --action_env TF_CUDA_VERSION="13.0.3" \
  --action_env TF_CUDA_COMPUTE_CAPABILITIES="12.0" \
  --define=using_rocm=false \
  --incompatible_disallow_empty_glob=false \
  //:xla_compute