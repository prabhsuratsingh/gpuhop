#!/usr/bin/env bash
# Isaac Lab instance setup. Runs once per instance from the project dir; re-runs only when this file,
# requirements.txt or requirements.lock.txt change (`gpuhop rebootstrap --force` reruns it anyway).
set -euo pipefail

echo "--- apt packages (headless Kit runtime libs, tools)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  git curl wget ca-certificates gnupg tmux htop pciutils build-essential \
  libglu1-mesa libxrender1 libxt6 libxi6 libsm6 libice6 libxrandr2 libxinerama1 libxcursor1 \
  libxkbcommon0 libegl1 libgl1 libvulkan1 vulkan-tools >/dev/null

echo "--- Vulkan ICD"
# The NVIDIA container toolkit mounts the Vulkan driver when NVIDIA_DRIVER_CAPABILITIES includes graphics, but
# some hosts ship no ICD manifest. Kit needs it even headless. Exactly one manifest must exist: two make Vulkan
# enumerate the same GPU twice.
FALLBACK_ICD=/usr/share/vulkan/icd.d/nvidia_icd.json
if [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
  if [ -f $FALLBACK_ICD ] && grep -q '"api_version": "1.3" } }' $FALLBACK_ICD; then rm -f $FALLBACK_ICD; echo "removed duplicate fallback ICD"; fi
elif [ ! -f $FALLBACK_ICD ]; then
  mkdir -p /usr/share/vulkan/icd.d
  cat > $FALLBACK_ICD <<'EOF'
{ "file_format_version": "1.0.0", "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version": "1.3" } }
EOF
  echo "wrote fallback nvidia_icd.json"
fi
vulkaninfo --summary 2>/dev/null | grep -E 'deviceName' || echo "WARNING: vulkaninfo found no device"

if [ "${GPUB_INSTALL_NSYS:-0}" = "1" ] && ! command -v nsys >/dev/null 2>&1; then
  echo "--- Nsight Systems CLI (not fatal)"
  {
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/ck.deb \
      && dpkg -i /tmp/ck.deb >/dev/null \
      && apt-get update -qq \
      && apt-get install -y -qq --no-install-recommends cuda-nsight-systems-12-8 >/dev/null \
      && NSYS_BIN=$(find /opt/nvidia /usr/local/cuda* -type f -name nsys -path '*/bin/*' 2>/dev/null | sort -V | tail -1) \
      && [ -n "$NSYS_BIN" ] && ln -sf "$NSYS_BIN" /usr/local/bin/nsys
  } || echo "WARNING: nsys install failed"
fi

echo "--- python packages"
python -m pip install -q --upgrade pip
REQ=requirements.lock.txt
[ -s "$REQ" ] || { echo "no lock file yet -> loose pins (run 'gpuhop freeze' on a good instance)"; REQ=requirements.txt; }
python -m pip install -q -r "$REQ" --extra-index-url https://pypi.nvidia.com

echo "--- sanity"
gcc --version | head -1   # Triton / torch.compile build their launcher stubs with it
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch cannot see a GPU"
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
EOF
