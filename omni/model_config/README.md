# 模型配置项自动加载使用说明
## 模型配置项定义
模型配置项统一定义在 `omni_npu/model_config/config_loader/loader.py` 的配置类中，如需添加，按照配置项功能放在对应的配置类中，注意初始值的设定，当模型对应的配置项json文件不包含该配置项时，设为默认值。

模型配置类分为三个：
1. `TaskConfig`: 任务相关的一些配置项，如当前模型的类型、当前执行的硬件环境、节点属性。
2. `ModelParallelConfig`: 推理时并行策略相关配置项，注意框架侧可获取的并行配置框架侧获取。
3. `ModelOperatorOptConfig`: 算子特性相关配置项。

三个模型配置类统一到`ModelExtraConfig`类中，若存在部分配置项存在冲突的情况，在对应的配置类的`__post_init__`中进行校验，并提供提示信息。

## 模型配置项加载时机
模型配置项的加载时机是在`npu_worker.py`的`init_device`函数中调用`loader.py`的对外暴露接口为`load_model_extra_config`，并传入框架端的config参数。
```python
   # Initialize the model best practice configs.
    load_model_extra_config(self.model_config, self.vllm_config, self.scheduler_config)
```
## 关于新增模型配置项
对于新增的模型配置项，需要先在`loader.py`的对应配置类上添加对应的配置项，请注意默认方式，非必须打开的配置项默认关闭，调用方式如下：
```python
from omni.models.config_loader.loader import model_extra_config
model_extra_config.operator_opt_config.xxxx
```
## 关于新增模型的配置项json
假如需要新增模型配置项json文件，需要以下几个步骤：
1. 检查当前模型在`match_hf_configs.json`中是否登记，若未登记，先登记；
对于新增模型，若新增模型需要新增配置项json文件，必须在v1\models\config文件夹下的`match_hf_configs.json`进行登记。
登记方法为：添加模型权重文件中的config.json文件上的架构相关的属性，用于匹配对应的模型类型。其中一级json对象的key一一对应`best_practice_configs.json`中的model字段，如`"qwen-235B"`、`"kimi-k2"`。
`match_hf_configs.json`格式如下：
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
2. 确定新增配置文件的使用场景
当前的模型配置文件分为模型最优配置文件和用户自定义配置文件，他们有以下不同：
- 最优配置文件是指在相同模型在相同部署形态下，根据不同性能目标所需要开启的模型配置，当前可分为`high_throughout`和`low_latency`两类，其中也可以根据性能的不同级别设置模型配置文件。
因此，在新增模型最优配置文件时，必须放在`high_throughout`和`low_latency`的对应路径下面。
其中路径默认为`high_throughout`，`low_latency`需要使用`ADDITIONAL_CONFIG`进行传入设置，传入方式如下：
    ```yaml
    ADDITIONAL_CONFIG='{"enable_low_latency":true}'
    ```
- 用户自定义配置文件是为了实现模型配置项的灵活使用设置的，可用于开发调测、性能无关问题的规避等场景，这些文件需要环境变量`CUSTOM_MODEL_CONFIG_PATH`控制，若打开，则优先使用自定义模型配置。
注意，`CUSTOM_MODEL_CONFIG_PATH`给定的是相对路径，必须在v1\models\config路径下面。

3. 检查对应文件下的`best_practice_configs.json`是否有和新增配置**相同的运行平台和量化类型**，若有，在对应的json对象中的configs内新增对应部署形态的配置文件路径，若无，参考其他json对象，新增对应`model_type`、`hardware_platform`、`quant_type`字段的json对象。
`best_practice_configs.json`格式如下：
    ```json
    {
        "model": "deepseek_v3",
        "hardware": "A3",
        "precision": "w8a8c16",
        "configs":{
            "1P1D": {
                "prefill_config_file": "deepseek/ds_r1_w8a8c16_a3_1p1d_p.json",
                "decode_config_file": "deepseek/ds_r1_w8a8c16_a3_1p1d_d.json"
            },
            "4P1D":{
                "prefill_config_file": "deepseek/ds_r1_w8a8c16_a3_4p1d_p.json",
                "decode_config_file": "deepseek/ds_r1_w8a8c16_a3_4p1d_d.json"
            }
        }
    },
    ```
    **注意**，为了减少配置文件冗余的情况，在UT测试中加入了配置文件的校验，**要求每个配置文件加载的配置类对象是唯一的**，假如两个配置文件加载后的配置类对象是一致的，会在UT中拦截。

4. 将新增的对应配置文件加入到**指定模型路径**下。



