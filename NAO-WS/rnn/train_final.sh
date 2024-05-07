source /home/cs20b057/venv/bin/activate
nvidia-smi
export PYTHONPATH=./:$PYTHONPATH
MODEL_DIR=models/
LOG_DIR=logs
DATA_DIR=data/penn

mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR

fixed_arc="0 2 1 2 2 1 3 3 4 1 2 0 3 2 2 1 6 2 7 1 6 0"

python train.py \
  --data=$DATA_DIR \
  --save=$MODEL_DIR \
  --epochs=4000 \
  --arch="$fixed_arc" 2>&1 | tee -a $LOG_DIR/train.log
