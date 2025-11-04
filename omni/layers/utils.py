# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
import torchair as tng

AIC_CORE_NUM_910C = 24
AIV_CORE_NUM_910C = 48
AIC_CORE_NUM_910B = 24
AIV_CORE_NUM_910B = 48

class ConditionalTNGScope:
    def __init__(
                self,
                multi_stream: bool = False,
                stream_id: str = '',
                super_kernel: bool = False,
                scope: str = '',
                options: str = 'stream-fusion=1', 
                core_num: str = ''
    ):
        self.multi_stream = multi_stream
        self.stream_id = stream_id
        self.super_kernel = super_kernel
        self.scope = scope
        self.options = options
        self.core_num = core_num
        self.contexts = self._build_contexts()
    
    def _build_contexts(self):
        contexts = []
        
        # super_kernel
        if self.super_kernel:
            contexts.append(tng.scope.super_kernel(self.scope, self.options))
        
        # multi_stream
        if self.multi_stream and self.stream_id:
            contexts.append(tng.scope.npu_stream_switch(self.stream_id))
        
        # limit_core
        if self.core_num:
            core_nums_list = self.core_num.split('|') 
            aic_core_num = int(core_nums_list[0])
            aiv_core_num = int(core_nums_list[1])
            if not self.stream_id:
                contexts.append(tng.scope.limit_core_num(aic_core_num, aiv_core_num))
            else:
                if os.getenv("ASCEND_PLATFORM", "A3") == "A2":
                    aic_core_total = AIC_CORE_NUM_910B
                    aiv_core_total = AIV_CORE_NUM_910B
                else:
                    aic_core_total = AIC_CORE_NUM_910C
                    aiv_core_total = AIV_CORE_NUM_910C
                contexts.append(tng.scope.limit_core_num(aic_core_total - aic_core_num, aiv_core_total - aiv_core_num))

        return contexts
    
    def __enter__(self):
        self.entered = []
        for ctx in self.contexts:
            entered_obj = ctx.__enter__()
            self.entered.append(entered_obj)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for ctx in reversed(self.contexts):
            ctx.__exit__(exc_type, exc_val, exc_tb)

