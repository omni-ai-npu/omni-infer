from typing import Optional

from vllm.v1.outputs import ModelRunnerOutput
from vllm.logger import logger
from vllm.v1.request import Request
from omni.accelerators.pd.cc_connector import CcConnectorWorker
from vllm.v1.core.sched.scheduler import Scheduler


def _update_waiting_for_remote_kv(self, request: Request) -> bool:
    assert self.connector is not None
    finished_recving_kv_reqs = {
        req_id: (int(num_success_tokens), item)
        for item in self.finished_recving_kv_req_ids
        for req_id, num_success_tokens in [item.split(CcConnectorWorker.SPLITTER)]
    }

    if finished_recving_kv_reqs:
        logger.debug(f"[EMS] finished_recving_kv_reqs: {finished_recving_kv_reqs}.")
    if request.request_id not in finished_recving_kv_reqs:
        return False
    
    num_computed_tokens = finished_recving_kv_reqs[request.request_id][0]
    
    if self.kv_cache_manager.enable_caching:
        self.kv_cache_manager.single_type_manager.cache_blocks(
            request,
            self.kv_cache_manager.req_to_block_hashes[request.request_id],
            num_computed_tokens,
        )

    # Update the request state for scheduling.
    request.num_computed_tokens = num_computed_tokens

    # Return that we are ready.
    self.finished_recving_kv_req_ids.remove(finished_recving_kv_reqs[request.request_id][1])
    return True

Scheduler._update_waiting_for_remote_kv = _update_waiting_for_remote_kv
print("++++++++++++++++++++++patch_ems++++++++++++++++++++++++++++")