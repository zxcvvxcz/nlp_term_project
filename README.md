# nlp_term_project
Codes for training KoBERT(https://huggingface.co/monologg/kobert) with contrastive losses on KoBEST dataset(https://huggingface.co/datasets/skt/kobest_v1).   
For HellaSwag and COPA, use kobert_mc_no_trainer.py.   
For BoolQ, WiC, and SentiNeg, use kobert_classification_no_trainer.py.   
utils.py includes datasets and loss functions for contrastive learning on each dataset.   
To install requirements, use requirements.txt as below.   
```
pip install -r requirements.txt
```
Huggingface tokenizer for KoBERT is erroneous. Instead, the tokenizer from kobert-transformers is used in experiments.

Example code
```
// mc, cross entropy only
python kobert_mc_no_trainer.py \
  --model_name_or_path monologg/kobert \
  --dataset_name skt/kobest_v1 --dataset_config_name [hellaswag|copa] \
  --output_dir mc \
  --per_device_train_batch_size 16 \
  --weight_decay 0.01 \
  --num_train_epochs 10 \
  --learning_rate 1e-5 \
  --max_length 256 \
  --patience 10
  
// mc, contrastive learning
cl_method = [cl|scl] // cl: unsupervised contrastive learning, scl: supervised contrastive learning
python kobert_mc_no_trainer.py \
  --model_name_or_path monologg/kobert \
  --dataset_name skt/kobest_v1 --dataset_config_name [hellaswag|copa] \
  --output_dir mc \
  --per_device_train_batch_size 16 \
  --weight_decay 0.01 \
  --cl_method $cl_method \
  --alpha 0.2 \
  --temperature 0.3 \
  --num_train_epochs 10 \
  --learning_rate 1e-5 \
  --max_length 256 \
  --patience 10


// classification, cross entropy
python kobert_classification_no_trainer.py \
    --model_name_or_path monologg/kobert \
    --task_name [boolq|wic|sentineg] \
    --output_dir classification \
    --per_device_train_batch_size 8 \
    --max_length 256 \
    --num_train_epochs 10 \
    --weight_decay 0.01 \
    --learning_rate 5e-6 \
    --patience 10

// classification, contrastive learning
cl_method = [cl|scl] // cl: unsupervised contrastive learning, scl: supervised contrastive learning
python kobert_classification_no_trainer.py \
    --model_name_or_path monologg/kobert \
    --task_name [boolq|wic|sentineg] \
    --output_dir classification \
    --per_device_train_batch_size 8 \
    --max_length 256 \
    --num_train_epochs 10 \
    --weight_decay 0.01 \
    --cl_method $cl_method \
    --learning_rate 5e-6 \
    --alpha 0.2 \
    --temperature 0.3 \
    --patience 10
    
// multi gpu
accelerate launch --num_processes 2 --gpu_ids 0,1 \
    kobert_classification_no_trainer.py \
    --model_name_or_path monologg/kobert \
    --task_name $dataset \
    --report_to all \
    --output_dir classification \
    --per_device_train_batch_size 8 \
    --max_length 256 \
    --num_train_epochs 10 \
    --weight_decay 0.01 \
    --cl_method $cl_method \
    --learning_rate 5e-6 \
    --alpha 0.2 \
    --temperature 0.3 \
    --patience 10
```