#!/bin/bash

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

NGPU=${NGPU:-4}

STYLEGAN_REPO_DIR="$DIR/../../"
SCRIPT="$STYLEGAN_REPO_DIR/generate_init.py"

python $SCRIPT  --outdir="$DIR/same_G" --gpus=$NGPU --res 128 \
 --init_seed=997711 --sampling_seeds 777778-877777 --no-auto-outdir-folder \
 --reinit-sample-interval -1 --save_ty raw_tensor

python "$DIR/compute_stats.py" $DIR/same_G

# init seeds below are set to avoid collision

python $SCRIPT  --outdir="$DIR/10_per_G" --gpus=$NGPU --res 128 \
 --init_seed=99997711 --sampling_seeds 777778-877777 --no-auto-outdir-folder \
 --reinit-sample-interval 10 --save_ty raw_tensor

python "$DIR/compute_stats.py" $DIR/10_per_G

python $SCRIPT  --outdir="$DIR/1_per_G" --gpus=$NGPU --res 128 \
 --init_seed=99997711 --sampling_seeds 777778-877777 --no-auto-outdir-folder \
 --reinit-sample-interval 1 --save_ty raw_tensor

python "$DIR/compute_stats.py" $DIR/1_per_G

