"""
ModelArts v2 在线服务创建客户端

流程:
    1. 先用 convert_body.py 把服务快照 JSON 转换成正确请求体
       (转换: python convert_body.py <快照json>)
    2. 再执行本脚本发起创建:
       python modelarts_client.py <正确请求体json路径>   ← 必传

凭证(环境变量, 不写入代码):
    HW_AK / HW_SK / HW_ST(security token; 临时凭证必填)

可覆盖环境变量:
    MODELARTS_REGION(默认 cn-east-4) / MODELARTS_PROJECT_ID /
    MODELARTS_WORKSPACE_ID / MODELARTS_ENTERPRISE_PROJECT_ID
"""

import json
import os
import sys
import hashlib
import hmac
import datetime
import urllib.request
import urllib.parse
import urllib.error

# ---------------------------------------------------------------- 常量
REGION = os.environ.get("MODELARTS_REGION", "cn-east-4")
HOST = "modelarts.%s.myhuaweicloud.com" % REGION
PROJECT_ID = os.environ.get(
    "MODELARTS_PROJECT_ID", "c7a4a765074e442099cdab43637ae7a3")
WORKSPACE_ID = os.environ.get(
    "MODELARTS_WORKSPACE_ID", "1d670a16e0b34acbb27975dfec4d3790")
ENTERPRISE_PROJECT_ID = os.environ.get(
    "MODELARTS_ENTERPRISE_PROJECT_ID",
    "095a4d22-3bc1-4501-844c-2309ed0e76c2")  # aifm-infra-infer_MA_eps
PATH = "/v2/%s/services" % PROJECT_ID


# ---------------------------------------------------------------- 签名工具
def encode_rfc3986(s: str) -> str:
    return urllib.parse.quote(s, safe="-_.~")


def hmac_sha256_hex(key: bytes, msg: str) -> str:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).hexdigest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_auth(method: str, path: str, body: bytes, extra_headers=None):
    """构造 SDK-HMAC-SHA256 签名的 Authorization 与 x-sdk-date。

    注意: 该 APIG 网关验签时会把不带尾斜杠的规范 URI 补成带 '/' 的形式,
    因此规范 URI 需补尾斜杠; Content-Type 头不能带 charset。
    """
    ak = os.environ["HW_AK"]
    sk = os.environ["HW_SK"]
    st = os.environ.get("HW_ST", "")
    x_sdk_date = datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # 规范 URI: 逐段编码并补尾斜杠
    canonical_uri = "/".join(encode_rfc3986(urllib.parse.unquote(seg))
                             for seg in path.split("/"))
    if not canonical_uri.endswith("/"):
        canonical_uri += "/"

    # 规范头(发送的头部全部纳入签名)
    hdrs = {"host": HOST, "x-sdk-date": x_sdk_date,
            "content-type": "application/json"}
    if st:
        hdrs["x-security-token"] = st
    if extra_headers:
        for k, v in extra_headers.items():
            hdrs[k.lower()] = v

    signed_hdr_items = sorted(
        (name, value) for name, value in hdrs.items() if value is not None)
    canonical_headers = "".join(
        "%s:%s\n" % (name, " ".join(value.split()))
        for name, value in signed_hdr_items)
    signed_headers = ";".join(name for name, _ in signed_hdr_items)

    canonical_request = "\n".join([
        method.upper(), canonical_uri, "", canonical_headers,
        signed_headers, sha256_hex(body)])
    string_to_sign = "\n".join([
        "SDK-HMAC-SHA256", x_sdk_date,
        sha256_hex(canonical_request.encode("utf-8"))])
    signature = hmac_sha256_hex(sk.encode("utf-8"), string_to_sign)
    auth = "SDK-HMAC-SHA256 Access=%s, SignedHeaders=%s, Signature=%s" % (
        ak, signed_headers, signature)
    return auth, x_sdk_date

def signed_request(method: str, path: str, body: bytes = b"",
                   extra_headers=None, timeout: int = 180):
    """发送签名请求, 返回 (HTTP状态码, 响应文本)。"""
    auth, x_sdk_date = make_auth(method, path, body, extra_headers)

    url = "https://%s%s" % (HOST, path)
    req = urllib.request.Request(url, data=body or None, method=method.upper())
    req.add_header("Authorization", auth)
    req.add_header("X-Sdk-Date", x_sdk_date)
    req.add_header("Content-Type", "application/json")
    req.add_header("Host", HOST)
    if os.environ.get("HW_ST"):
        req.add_header("X-Security-Token", os.environ["HW_ST"])
    if extra_headers:
        for k, v in extra_headers.items():
            req.add_header(k, v)

    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except urllib.error.URLError as e:
        return 0, "网络错误: %s" % e.reason


# ---------------------------------------------------------------- 业务
def load_body(body_file):
    if not os.path.isfile(body_file):
        raise FileNotFoundError("请求体文件不存在: %s" % body_file)
    with open(body_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if "group_configs" not in raw:
        raise ValueError(
            "输入不是规范请求体(缺顶层 group_configs): %s\n"
            "请先用转换脚本生成: python convert_body.py <服务快照json>"
            % body_file)
    return raw


def create_service(body_file: str):
    """读取规范请求体并 POST 创建在线服务, 返回 (status, text)。"""
    raw = load_body(body_file)
    body = json.dumps(raw, ensure_ascii=False).encode("utf-8")
    extra = {"X-Workspace-Id": WORKSPACE_ID,
             "X-Enterprise-Project-ID": ENTERPRISE_PROJECT_ID}
    return signed_request("POST", PATH, body, extra_headers=extra)


# ---------------------------------------------------------------- 入口
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        print("用法: python modelarts_client.py <规范请求体json路径>",
              file=sys.stderr)
        return 2
    body_file = argv[0]

    if not (os.environ.get("HW_AK") and os.environ.get("HW_SK")):
        print("缺少环境变量 HW_AK / HW_SK (临时凭证还需 HW_ST)", file=sys.stderr)
        return 2

    status, text = create_service(body_file)
    print("HTTP %s" % status)
    print(text)

    try:
        obj = json.loads(text) if text else {}
        ok = (200 <= status < 300) and bool(obj.get("id"))
    except Exception:
        ok = False
    if ok:
        print("\n创建成功: id=%s name=%s status=%s"
              % (obj["id"], obj.get("name"), obj.get("status")))
    return 0 if (200 <= status < 300) else 1


if __name__ == "__main__":
    sys.exit(main())
