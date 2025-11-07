import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Type, Union, Tuple, Callable
from collections import defaultdict
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from http import HTTPStatus
from vllm.entrypoints.openai.protocol import ErrorResponse


TYPE_MAPPING = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict
}
DEFAULT_MAX_MODEL_LEN = 8192

class BaseValidator(ABC):
    def __init__(self,
                 param_name: str,
                 error_msg: Optional[str] = None
                 ):
        self.param_name = param_name
        self.error_msg = error_msg

    def validate(self, value: Any) -> Optional[str]:
        pass

    def validate_json(self, value: Any) -> Optional[str]:
        pass


class NestedBaseValidator(BaseValidator):
    def __init__(
        self,
        param_name: str,
        error_msg: Optional[str] = None,
        subfield: list[str] = [],
        checker_condition = None,
        checker: Callable[[str, Any], Tuple[Optional[str], Optional[Any]]] | None = None,
        skip_check_subfield: list = []
    ):
        super(NestedBaseValidator, self).__init__(param_name, error_msg)
        self.subfield = subfield
        self.checker_condition = checker_condition
        self.checker = checker
        self.skip_check_subfield = skip_check_subfield
    
    def validate(self, value):
        if not self.subfield:
            return None
        return self.check_field(value, self.param_name)
    
    def check_field(self, value, param_name: str) -> Optional[str]:
        if isinstance(value, dict):
            return self.check_dict_subfield(value, param_name)
        if isinstance(value, list):
            return self.check_list_subfield(value, param_name)
        return None
    
    def check_dict_subfield(self, value, cur_param: str) -> Optional[str]:
        if cur_param in self.skip_check_subfield:
            return None
        for name, val in list(value.items()):
            sub_cur_param = f'{cur_param}.{name}'
            if self.checker_condition and self.checker_condition(sub_cur_param):
                if self.checker:
                    if err_str := self.checker(sub_cur_param, value):
                        return err_str
            elif isinstance(val, (list, dict)):
                err_str = self.check_field(val, sub_cur_param)
                if err_str:
                    return err_str
        return None
    
    def check_list_subfield(self, value, cur_param: str) -> Optional[str]:
        for val in value:
            if isinstance(val, dict):
                err_str = self.check_dict_subfield(val, cur_param)
                if err_str:
                    return err_str
        return None


class SupportedValidator(NestedBaseValidator):
    def __init__(
            self,
            param_name: str,
            error_msg: Optional[str] = None,
            subfield: list = [],
            skip_check_subfield: list = []
    ):
        def checker_condition(param_name: str):
            return param_name not in self.subfield
        
        def checker(param_name: str, value: Any):
            value.pop(param_name.split('.')[-1])
            return None
        
        super().__init__(param_name, error_msg, subfield, checker_condition, checker, skip_check_subfield)


class NestedValueValidator(NestedBaseValidator):
    def __init__(
        self,
        param_name: str,
        error_msg: Optional[str] = None,
        subfield: list[str] = [],
        target_values: list[str] = []
    ):
        def checker_condition(param_name: str):
            return param_name in self.subfield
        
        def checker(param_name: str, value: Any):
            if value[param_name.split('.')[-1]] not in target_values:
                return (f'{param_name} only support the value in {target_values}', None)
            return None
        
        super().__init__(param_name, error_msg, subfield, checker_condition, checker)

class IncompatibilityValidator(BaseValidator):
    def __init__(self,
                 param_name: str,
                 error_msg: Optional[str] = None,
                 subfield: list[str] = []
                 ):
        super().__init__(param_name, error_msg)
        self.subfield = subfield

    def validate_json(self, request_json):
        for param_name in self.subfield:
            request_json.pop(param_name, None)


class RangeValidator(BaseValidator):
    def __init__(self,
                 param_name: str,
                 error_msg: Optional[str] = None,
                 min_val: Union[float, int, None] = None,
                 max_val: Union[float, int, None] = None,
                 type_: Union[tuple[Type, ...], None] = None
                 ):
        super().__init__(param_name, error_msg)
        self.min_val = min_val
        self.max_val = max_val
        self.type_ = type_
        
    def validate(self, value: Any) -> Optional[str]:
        if self.type_ and not isinstance(value, self.type_):
            return (self.error_msg or 
                    f"The type of `{self.param_name}` must belong to {[i.__name__ for i in self.type_]}, "
                    f"but got {type(value).__name__!r}")
        if not (self.min_val <= value <= self.max_val):
            return (self.error_msg or f"`{self.param_name}` must between {self.min_val} and {self.max_val}, "
                    f"but got {value}.")
        return None


class ValueValidator(SupportedValidator):
    def __init__(
            self,
            param_name: str,
            error_msg: Optional[str] = None,
            subfield: list = [],
            target_value: list = []
    ):
        super().__init__(param_name, error_msg, subfield)
        self.target_value = target_value
        self.error_msg = self.error_msg or f"`{self.param_name}` only support the value in {self.target_value}"

    def validate(self, value):
        if error := super().validate(value):
            return error
        if value not in self.target_value:
            return self.error_msg
        return None
    
    def validate_json(self, request_json):
        value = request_json[self.param_name]
        if error := super().validate(value):
            return error
        if value not in self.target_value:
            request_json.pop(self.param_name)
        return None


def create_validator(param_name: str, config: dict[str, Any]) -> Optional[BaseValidator]:
    validator_type = config.get("validator_type")
    
    if validator_type == "supported":
        return SupportedValidator(
            param_name=config.get("param_name", param_name),
            error_msg=config.get("error_msg"),
            subfield=config.get("subfield", []),
            skip_check_subfield=config.get("skip_check_subfield", [])
        )
    
    elif validator_type == "incompatibility":
        return IncompatibilityValidator(
            param_name=config.get("param_name", param_name),
            error_msg=config.get("error_msg"),
            subfield=config.get("subfield", [])
        )
    
    elif validator_type == "value":
        return ValueValidator(
            param_name=config.get("param_name", param_name),
            error_msg=config.get("error_msg"),
            subfield=config.get("subfield", []),
            target_value=config.get("target_value", [])
        )
    
    elif validator_type == "range":
        type_str = config.get("type_", [])
        if type_str and any(type_ not in TYPE_MAPPING for type_ in type_str):
            raise ValueError(f"Only supported type: {TYPE_MAPPING.keys()}")

        return RangeValidator(
            param_name=config.get("param_name", param_name),
            min_val=config.get("min_val"),
            max_val=config.get("max_val"),
            type_=tuple(map(TYPE_MAPPING.get, type_str))
        )
    elif validator_type == 'nested_value':
        return NestedValueValidator(
            param_name=config.get('param_name', param_name),
            error_msg=config.get('error_msg'),
            subfield=config.get('subfield', []),
            target_values=config.get('target_value', [])
        )
    
    else:
        raise ValueError(f"Unknown validator type: {validator_type}")


def load_validators_from_json(config_path: str) -> tuple[dict[str, BaseValidator], dict[str, BaseValidator]]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    validators = defaultdict(list)
    validators_json = defaultdict(list)
    
    # load validators
    for param_name, validator_config in config.get("validators", {}).items():
        if not isinstance(validator_config, list):
            validator_config = [validator_config]
        for cfg in validator_config:
            validator = create_validator(param_name, cfg)
            if validator:
                validators[param_name].append(validator)
    
    # load validators_json
    for param_name, validator_config in config.get("validators_json", {}).items():
        if not isinstance(validator_config, list):
            validator_config = [validator_config]
        for cfg in validator_config:
            validator = create_validator(param_name, cfg)
            if validator:
                validators_json[param_name].append(validator)
    
    return validators, validators_json


VALIDATORS, VALIDATORS_JSON = load_validators_from_json(os.getenv("VALIDATORS_CONFIG_PATH", ""))


class ValidateSamplingParams(BaseHTTPMiddleware):
    def create_error_response(self, status_code, error):
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                message=str(error),
                type="BadRequestError",
                code=status_code.value
            ).model_dump()
        )

    def replace_with_stars(self, text):
        return "*" * len(text)
    
    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and request.url.path in ("/v1/completions", "/v1/chat/completions"):
            body = await request.body()
            if not body:
                return await call_next(request)
            
            try:
                json_load = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                return await call_next(request)

            # Only True and False are allowed for "thinking" in chat_template_kwargs.thinking.
            chat_template_kwargs = json_load.get("chat_template_kwargs")
            if chat_template_kwargs is not None and isinstance(chat_template_kwargs, dict) and "thinking" in chat_template_kwargs \
                    and chat_template_kwargs["thinking"] not in {True, False}:
                chat_template_kwargs.pop("thinking")
            # Ignore guided decoding if parameter json_schema is missing.
            response_format = json_load.get("response_format")
            if response_format is not None and isinstance(response_format, dict) and "type" in response_format \
                and response_format["type"] == "json_schema":
                if response_format.get("json_schema", None) is None:
                    json_load.pop("response_format")
            # guided_decoding_backend is not supported
            if "guided_decoding_backend" in json_load:
                json_load.pop("guided_decoding_backend")
            request._body = json.dumps(json_load).encode("utf-8")

            if json_load.get("kv_transfer_params"):
                max_tokens = json_load.get("max_tokens", None)
                if not max_tokens:
                    json_load["max_tokens"] = int(os.getenv("DEFAULT_MAX_TOKENS", DEFAULT_MAX_MODEL_LEN))
                    request._body = json.dumps(json_load).encode("utf-8")

            if not VALIDATORS:
                return await call_next(request)
            
            status_code = HTTPStatus.BAD_REQUEST
            for param_name, value in list(json_load.items()):
                validators = VALIDATORS.get(param_name)
                if not validators:
                    json_load.pop(param_name)
                    continue
                for validator in validators:
                    if error := validator.validate(value):
                        return self.create_error_response(status_code, error)
                if validators := VALIDATORS_JSON.get(param_name):
                    for validator in validators:
                        if error := validator.validate_json(json_load):
                            return self.create_error_response(status_code, error)
            
            if os.environ.get("ROLE", "") == "prefill":
                json_load["max_tokens"] = 1
            request._body = json.dumps(json_load).encode("utf-8")

        if request.method == "GET" and request.url.path == "/v1/models":
            response = await call_next(request)
            chunk = await anext(response.body_iterator)
            chunk_json = json.loads(chunk.decode("utf-8"))

            if chunk_json is not None and len(chunk_json.get("data", [])) > 0 and chunk_json.get("data")[0].get("root"):
                chunk_json.get("data")[0]["root"] = self.replace_with_stars(chunk_json.get("data")[0].get("root"))

            new_json_str = json.dumps(chunk_json, ensure_ascii=False)
            new_chunk = new_json_str.encode("utf-8")

            return Response(
                content=new_chunk,
                headers={
                    "Content-Length": str(len(new_chunk)),
                    'content-type': 'application/json'
                }
            )

        return await call_next(request)