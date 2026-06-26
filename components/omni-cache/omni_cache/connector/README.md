# Connector Maintainer Notes

## Scope
This note describes the connector-side split after refactor and where to place future changes.

Primary entrypoint:
1. [connector.py](connector.py) - Main OmniCacheConnector class

Split worker modules:
1. [prefill/worker.py](prefill/worker.py) - PrefillConnectorWorker implementation
2. [decode/worker.py](decode/worker.py) - DecodeConnectorWorker implementation

Shared helper modules (in utils/):
1. [utils/helpers.py](utils/helpers.py) - connector helpers
2. [utils/metadata.py](utils/metadata.py) - connector models
3. [utils/settings.py](utils/settings.py) - connector settings
4. [utils/process_utils.py](utils/process_utils.py) - process utilities

## Responsibility Boundaries

### Entrypoint
File: [connector.py](connector.py)

Responsibilities:
1. Keep public connector-facing symbols stable.
2. Keep scheduler-side orchestration and API compatibility.

### Prefill worker
File: [prefill/worker.py](prefill/worker.py)

Responsibilities:
1. Prefill-side server startup and lifecycle.
2. Prefill finished-request accounting and delayed release handling.

### Decode worker
File: [decode/worker.py](decode/worker.py)

Responsibilities:
1. Decode-side async pull, queueing, and process/thread orchestration.
2. Decode-side h2d post-processing and receive-completion signaling.

### Shared helpers
Files (in utils/):
1. [utils/helpers.py](utils/helpers.py) - connector helpers
2. [utils/metadata.py](utils/metadata.py) - connector models
3. [utils/settings.py](utils/settings.py) - connector settings
4. [utils/process_utils.py](utils/process_utils.py) - process utilities

Responsibilities:
1. Keep pure helper/model/config/process logic centralized.
2. Avoid duplicating logic in worker modules.

## Extension Constraints

1. New prefill-only logic goes to [prefill/worker.py](prefill/worker.py).
2. New decode-only logic goes to [decode/worker.py](decode/worker.py).
3. Shared logic must be added to helper modules, not copy-pasted into both workers.
4. Keep entrypoint imports lazy where runtime side effects are possible.

## Related Design Docs

1. Global split constraints: [../../docs/REFACTOR_REPORT.md](../../docs/REFACTOR_REPORT.md)
2. Cache runtime patching (removed)
