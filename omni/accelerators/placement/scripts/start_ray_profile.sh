# export GLOO_SOCKET_IFNAME=enp67s0f5
# export TP_SOCKET_IFNAME=enp67s0f5
# export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

# export MM_ALL_REDUCE_OP_THRESHOLD=1000000
# export HCCL_OP_EXPANSION_MODE="AIV"
# export NUMEXPR_MAX_THREADS=192

# export ENABLE_MOE_EP=1

# export ASCEND_RT_VISIBLE_DEVICES=0,1
# ray start --head --num-gpus=2

# vim start_ray.sh
export GLOO_SOCKET_IFNAME=enp67s0f5
export TP_SOCKET_IFNAME=enp67s0f5
# export GLOO_SOCKET_IFNAME=eth4
# export TP_SOCKET_IFNAME=eth4
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

export VLLM_ENABLE_PROFILING=1
export VLLM_TORCH_PROFILER_DIR=/home/yjf/profiling

export MM_ALL_REDUCE_OP_THRESHOLD=1000000
export HCCL_OP_EXPANSION_MODE="AIV"
export NUMEXPR_MAX_THREADS=192
export ENABLE_MOE_EP=1
export DP_SIZE=4
# unset DP_SIZE

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ray start --head --num-gpus=8 --port=6377
