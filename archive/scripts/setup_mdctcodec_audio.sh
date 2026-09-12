#!/usr/bin/env bash
# Build Google's official ViSQOL v3.3.3 CLI and install the audio dependencies.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m pip install -r requirements-audio.txt
# audiotools pins protobuf 3.x, whereas the current W&B client needs >=4.21.
python -m pip install descript-audio-codec==1.0.0
python -m pip install 'protobuf>=5.29,<7'
if [[ ! -d outputs/mdctcodec_reference ]]; then
  git clone https://github.com/PB20000090/MDCTCodec.git outputs/mdctcodec_reference
  git -C outputs/mdctcodec_reference checkout 1aff8b6287f4e2e66511bd4bc46f3a6bcf96edaf
fi
if [[ -x outputs/visqol/bin/visqol ]]; then exit 0; fi
task_build_root="${TMPDIR:-/tmp}/laser-visqol-build"
mkdir -p "$task_build_root"
if [[ ! -d "$task_build_root/source" ]]; then
  git clone --depth 1 --branch v3.3.3 https://github.com/google/visqol.git "$task_build_root/source"
fi
curl -fLsS https://github.com/bazelbuild/bazel/releases/download/5.4.1/bazel-5.4.1-linux-x86_64 -o "$task_build_root/bazel"
chmod +x "$task_build_root/bazel"
# The 9.860.2 SourceForge tarball in ViSQOL's WORKSPACE is no longer served.
# Ubuntu's Armadillo headers replace that unavailable build dependency only.
if [[ ! -f /usr/include/armadillo ]]; then
  apt-get update
  apt-get install -y libarmadillo-dev
fi
mkdir -p "$task_build_root/armadillo/include"
ln -sfn /usr/include/armadillo "$task_build_root/armadillo/include/armadillo"
ln -sfn /usr/include/armadillo_bits "$task_build_root/armadillo/include/armadillo_bits"
cat > "$task_build_root/armadillo/WORKSPACE" <<'EOF'
workspace(name="armadillo_headers")
EOF
cat > "$task_build_root/armadillo/BUILD" <<'EOF'
cc_library(name="armadillo_header", hdrs=glob(["include/armadillo", "include/armadillo_bits/*.hpp"]), includes=["include/"], visibility=["//visibility:public"])
EOF
project_root="$PWD"
cd "$task_build_root/source"
"$task_build_root/bazel" --batch build :visqol -c opt --jobs=8 --override_repository="armadillo_headers=$task_build_root/armadillo"
mkdir -p "$project_root/outputs/visqol/bin/model"
cp bazel-bin/visqol "$project_root/outputs/visqol/bin/visqol"
cp model/libsvm_nu_svr_model.txt model/lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite "$project_root/outputs/visqol/bin/model/"
