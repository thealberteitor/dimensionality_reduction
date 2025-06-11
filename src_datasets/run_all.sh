#!/usr/bin/env bash

export R_LIBS_USER=~/R/libs
export PATH="$HOME/.local/bin:$PATH"

dirs=("Adult" "Mushroom" "Fashion_MNIST" "iris" "MNIST" "olivetti_faces" "Wine")


mkdir -p logs

for d in "${dirs[@]}"; do
  pushd "$d" > /dev/null
  for script in *.R; do
    base=${script%.R}
    echo ">>> Ejecutando $d/$script"
    nohup Rscript "$script" \
      > "../logs/${d}_${base}.log" 2>&1 </dev/null

    if [ $? -ne 0 ]; then
      echo "!! Error en $d/$script, mira logs/${d}_${base}.log"
    fi
  done
  popd > /dev/null
done

echo "FIN.  Resultados en: ~/TFM Estadística/logs."
