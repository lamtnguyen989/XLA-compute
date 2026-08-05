CUDA_VER=13.0.3
CUDA_COMPUTE_CAPABILITIES=12.0

bazel build -c opt --config=cuda \
  --repo_env TF_NEED_CUDA=1 \
  --repo_env TF_NEED_ROCM=0 \
  --action_env TF_CUDA_VERSION="$CUDA_VER" \
  --action_env TF_CUDA_COMPUTE_CAPABILITIES="$CUDA_COMPUTE_CAPABILITIES" \
  --define=using_rocm=false \
  --incompatible_disallow_empty_glob=false \
  //:xla_compute