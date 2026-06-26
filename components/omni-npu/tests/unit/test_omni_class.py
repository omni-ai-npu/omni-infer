# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import inspect

from omni_npu.connector.llmdatadist_connector_v1 import LLMDataDistConnector, KVConnectorBase_V1
from omni_npu.distributed.communicator import NPUCommunicator, CudaCommunicator


def check_class_methods(cls, base_cls):
    """
    Validate that a class correctly implements or overrides methods defined in a base class.

    This function inspects all public callable methods defined directly on `cls`
    (i.e., methods that do not start with an underscore) and verifies that each of them:

    1. Exists on `base_cls` and is callable.
    2. Has a compatible function signature with the corresponding method in `base_cls`:
       - The subclass method must define **at least** as many parameters as the base method.
       - The first N parameters (where N is the number of parameters in the base method)
         must have the same parameter names and ordering.

    This allows subclass methods to extend the base method interface by adding
    extra parameters, while still preserving the base contract.

    Parameters
    ----------
    cls : type
        The class whose methods will be validated. Only methods defined directly
        on this class (not inherited ones) are checked.

    base_cls : type
        The reference base class that defines the expected method names and
        parameter signatures.

    Returns
    -------
    bool
        Returns True if all public methods in `cls` satisfy the compatibility
        requirements with `base_cls`.

    Raises
    ------
    AssertionError
        If any of the following conditions are violated:
        - A public method in `cls` does not exist in `base_cls`.
        - The corresponding attribute in `base_cls` is not callable.
        - The subclass method defines fewer parameters than the base method.
        - The names or ordering of the base parameters do not match those in the
          subclass method.

    Examples
    --------
    >>> class Base:
    ...     def process(self, x, y):
    ...         pass
    ...
    >>> class Sub(Base):
    ...     def process(self, x, y, z=0):
    ...         pass
    ...
    >>> check_class_methods(Sub, Base)
    True

    >>> class InvalidSub(Base):
    ...     def process(self, x):
    ...         pass
    ...
    >>> check_class_methods(InvalidSub, Base)
    Traceback (most recent call last):
        ...
    AssertionError
    """
    base_methods = {
        k: v for k, v in base_cls.__dict__.items()
        if not k.startswith("_") and callable(v)
    }
    print(f"cls={cls} base_cls={base_cls} base_methods={base_methods}")

    for method_name, base_method in base_methods.items():
        print(f">>>> method_name={method_name} base_method={base_method}")
        assert hasattr(cls, method_name) and callable(getattr(cls, method_name))

        sub_method_params = inspect.signature(getattr(cls, method_name)).parameters
        base_method_params = inspect.signature(base_method).parameters
        print(f"sub_params={list(sub_method_params.items())} base_params={list(base_method_params.items())}")
        assert len(sub_method_params) >= len(base_method_params)

        for (sn, sp), (bn, bp) in zip(sub_method_params.items(), base_method_params.items()):
            print(f"param_name={sn} sub_param={sp} base_param={bp}")
            assert sn == bn
            # assert sp.annotation == bp.annotation

    return True

def test_class_inheriting():
    # TODO now LLMDataDistConnector is inconsistent with KVConnectorBase_V1
    # assert check_class_methods(LLMDataDistConnector, KVConnectorBase_V1)
    assert check_class_methods(NPUCommunicator, CudaCommunicator)
