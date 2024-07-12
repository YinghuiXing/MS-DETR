#!/bin/bash

# 定义函数，用于执行 Python 命令
run_python_command() {
    CUDA_VISIBLE_DEVICES=$1 python fusion_main.py --output_dir $output_dir --action test --resume $2 --exp_config $exp_config > /dev/null 2>&1
}

# 设置输出目录和配置文件路径
output_dir="/data/wangsong/results/23_9_26/exp12"
exp_config="exp_config/23_9_26/exp_12.yaml"

# checkpoint 路径部分
checkpoint_path="/data/wangsong/results/23_9_26/exp12/checkpoint"

# 总任务数
total_tasks=20
gpu_num=4

# 循环遍历五次
for i in $(seq 0 $((total_tasks / gpu_num - 1)))
do
    # 循环创建四个子进程
    for j in $(seq 0 $((gpu_num - 1)))
    do
        task_id=$((i * gpu_num+j))  # 计算任务编号
        run_python_command $j "$checkpoint_path$(printf "%04d" $task_id).pth" &
    done

    # 等待这四个任务完成
    wait

    # 打印提示信息
    echo "Four tasks in iteration $((i + 1)) have completed."
done

# 所有任务完成后的后续代码...