# 模型配置项自动加载使用说明
## 模型配置项定义
模型配置项统一定义在omni\models\config_loader\loader.py的配置类中，如需添加，按照配置项功能放在对应的配置类中，注意初始值的设定，当模型对应的配置项json文件不包含该配置项时，设为默认值。

模型配置类分为三个：
1. `TaskConfig`: 任务相关的一些配置项，如当前模型的类型、当前执行的硬件环境、节点属性。
2. `ModelParallelConfig`: 推理时并行策略相关配置项，注意框架侧可获取的并行配置框架侧获取。
3. `ModelOperatorOptConfig`: 算子特性相关配置项。

三个模型配置类统一到`ModelExtraConfig`类中，若存在部分配置项存在冲突的情况，在`ModelExtraConfig`类的`__post_init__`中进行校验，并提供提示信息。

## 模型配置项加载时机
模型配置项先初始化为默认值，在`npu_worker.py`的`NPUWorker`的`_init_model_best_practice_configs`方法获取`TaskConfig`中对应的任务相关的一些配置项，最后通过`call_config_updater`对模型配置项进行更新，调用接口实现如下：
```python
   def _init_model_best_practice_configs(self):

        hardware_platform = "A2" if torch_npu.npu.get_device_name(0).startswith("Ascend910B") else "A3"
        is_pd_disaggregation = False
        is_prefill_node = None
        if os.getenv('ROLE', None):
            is_pd_disaggregation = True
            is_prefill_node = True if os.getenv('ROLE', None)=='prefill' else False
        enable_chunked_prefill = self.scheduler_config.enable_chunked_prefill
        enable_omni_placement = self.vllm_config.additional_config.get("enable_omni_placement", False)

        max_num_reqs = self.scheduler_config.max_num_seqs
        self.decode_gear_list = self.vllm_config.npu_compilation_config.decode_gear_list
        if self.decode_gear_list is None:
            self.decode_gear_list = []
            self.decode_gear_list.append(max_num_reqs if not self.vllm_config.speculative_config else max_num_reqs * \
                                            (1 + self.vllm_config.speculative_config.num_speculative_tokens))
        call_config_updater(
            config_updater_name = 'update_task_config',
            hf_config = self.model_config.hf_config,
            hardware_platform = hardware_platform,
            is_pd_disaggregation = is_pd_disaggregation,
            is_prefill_node = is_prefill_node,
            prefill_node_num = int(os.getenv("PREFILL_POD_NUM", 1)),
            decode_node_num = int(os.getenv("DECODE_POD_NUM", 1)),
            enable_chunked_prefill = enable_chunked_prefill,
            enable_omni_placement = enable_omni_placement,
            decode_gear_list=self.decode_gear_list,
            enable_graph_mode=self.enable_torchair_graph_mode
        )
```
其中，`call_config_updater`通过注册回调函数的方式实现，便于后续实现新的配置项更新策略。`config_updater_name`表示使用的配置项更新函数名，在omni\models\config_loader\loader.py中实现注册接口。其他参数是需要更新的不定参数，通过`**kwargs`传入。
## 关于enable_omni_placement场景的使用
`enable_omni_placement`配置项都需要通过cli或ansible传入，传入途径为`run_vllm_server_prefill_cmd`或`run_vllm_server_decode_cmd`的`addtional_config`，参数名分别`enable_omni_placement`，需要传入的对应的打开的指令`True`或`False`，placement的其他相关配置参数也需要`addtional_config`进行传入。参入参数格式如下：
```yaml
ADDITIONAL_CONFIG='{"graph_model_compile_config": {"level":1}, "enable_omni_placement":true,"omni_placement_config":{"pattern_path":"../../omni/accelerators/placement/patterns/base_patterns/DSV3_baseline_32_devices_58_MoE_Layers.npy","max_moe_layer_num":58,"enable_dynamic":true,"max_redundant_per_expert":1,"max_redundant_per_rank":0,"enable_rank_round_robin":true,"enable_dump":false,"dump_dir":"/home/profiling/dump_data"}}'
```
注意，`enable_omni_placement`设为`True`，其他参数必须一起传入。
## 关于新增模型配置项
对于新增的模型配置项，需要先在`loader.py`的对应配置类上添加对应的配置项，请注意默认方式，非必须打开的配置项默认关闭，调用方式如下：
```python
from omni.models.config_loader.loader import model_extra_config
model_extra_config.operator_opt_config.xxxx
```
## 关于新增模型的配置项json
对于新增模型，若新增模型需要新增配置项json文件，需要在omni\models\config文件夹下的`match_hf_configs.json`和`best_practice_configs.json`进行维护。`match_hf_configs.json`需要添加模型权重文件中的config.json文件上的一些属性，用于匹配对应的模型类型。`match_hf_configs.json`格式如下：
```json
{
    "deepseek_v3":{
        "model_type": "deepseek_v3",
        "hidden_size": 7168,
        "num_attention_heads": 128,
        "max_position_embeddings": 163840,
        "vocab_size": 129280,
        "intermediate_size": 18432,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "moe_intermediate_size": 2048
    },
    "qwen-235B":{
        "model_type": "qwen3_moe",
        "hidden_size": 4096,
        "num_attention_heads": 64,
        "max_position_embeddings": 262144,
        "vocab_size": 151936,
        "intermediate_size": 12288,
        "n_routed_experts": 128,
        "n_shared_experts": null,
        "moe_intermediate_size": 1536
    },
    "kimi-k2":{
        "model_type": "kimi-k2",
        "hidden_size": 7168,
        "num_attention_heads": 64,
        "max_position_embeddings": 131072,
        "vocab_size": 163840,
        "intermediate_size": 18432,
        "n_routed_experts": 384,
        "n_shared_experts": 1,
        "moe_intermediate_size": 2048
    }
}
```
`best_practice_configs.json`需要维护task_config上的部分静态配置项，包括`model_type`、`hardware_platform`、`quant_type`和`is_pd_disaggregation`。
`best_practice_configs.json`格式如下：
```json
   {
      "model": "deepseek_v3",
      "hardware": "A3",
      "precision": "w8a8",
      "prefill_node_num": 2,
      "decode_node_num": 1,
      "pd_disaggregation": true,
      "prefill_config_file": "deepseek_v3_a3_prefill_w8a8_2p1d.json",
      "decode_config_file": "deepseek_v3_a3_decode_w8a8_2p1d.json"
   },
   {
      "model": "deepseek_v3",
      "hardware": "A2",
      "precision": "bf16",
      "prefill_node_num": 2,
      "decode_node_num": 1,
      "pd_disaggregation": true,
      "prefill_config_file": "deepseek_v3_a2_prefill_bf16_2p1d.json",
      "decode_config_file": "deepseek_v3_a2_decode_bf16_2p1d.json"
   },
   {
      "model": "kimi-k2",
      "hardware": "A3",
      "precision": "w4a8",
      "prefill_node_num": 2,
      "decode_node_num": 1,     
      "pd_disaggregation": true,
      "prefill_config_file": "kimi_k2_a3_prefill_w4a8_2p1d.json",
      "decode_config_file": "kimi_k2_a3_decode_w4a8_2p1d.json"
   },
```
注意，`prefill_config_file`和`decode_config_file`分别是P节点和D节点对应的最佳配置项文件，任何需要应用或测试模型配置项的都需要添加或修改模型配置文件，禁止修改配置类上的默认值。



