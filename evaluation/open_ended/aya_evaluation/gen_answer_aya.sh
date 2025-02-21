gpus=(0 1 2 3)                    #a list of gpus to enable parallel generation on multiple gpus

lora_pathes=(
    "../../../models/FedAya/FedAya_25000_fedavg_c38s4_i10_b4a4_l1024_r16a32_20250221134749/checkpoint-10"
)                                           # a list of lora paths

for ((i=0; i<${#lora_pathes[@]}; i++)); do

    lora_path=${lora_pathes[$i]}

    gpu=${gpus[$i]}

    CUDA_VISIBLE_DEVICES=$gpu python gen_answer_aya.py \
    --base_model_path "Qwen/Qwen2-0.5B" \
    --lora_path $lora_path \
    --language_list English Standard_Arabic Russian Simplified_Chinese Portuguese French Spanish Telugu &
done

wait