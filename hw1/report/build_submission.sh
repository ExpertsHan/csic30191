#!/bin/bash
# 用法：bash report/build_submission.sh <學號>
set -euo pipefail
ID=${1:?請給學號，例如 bash report/build_submission.sh 313551000}
HW=$(cd "$(dirname "$0")/.." && pwd)
cd "$HW"
if grep -qE '{{[A-Z_]+}}' report/report.html; then echo "report.html 還有未填的 {{...}}："; grep -o '{{[A-Z_]*}}' report/report.html | sort -u; exit 1; fi
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --no-pdf-header-footer \
  --virtual-time-budget=15000 --print-to-pdf="$HW/report/report.pdf" "$HW/report/report.html" 2>/dev/null
rm -rf submit "${ID}_lab1.zip" && mkdir submit
cp report/report.pdf submit/
rsync -a --exclude 'data' --exclude 'results' --exclude 'samples' --exclude 'afhq_inception_v3.ckpt' \
  --exclude '__pycache__' --exclude '.ipynb_checkpoints' --exclude '.pytest_cache' --exclude '.DS_Store' \
  --exclude 'report_*.png' --exclude '*.ckpt' \
  Lab1-DDPM/2d_plot_diffusion_todo Lab1-DDPM/image_diffusion_todo submit/
(cd submit && zip -qr "../${ID}_lab1.zip" report.pdf 2d_plot_diffusion_todo image_diffusion_todo)
unzip -l "${ID}_lab1.zip"
unzip -l "${ID}_lab1.zip" | grep -E '\.ckpt|/data/|/results/|/samples/' && echo "!! 夾帶了不該交的檔案" || echo "OK：沒有夾帶 ckpt / data / results / samples"
