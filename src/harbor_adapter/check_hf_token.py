import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

HUB = "https://huggingface.co"
ENV_FILE = Path(os.environ.get("INFERENCE_BENCH_ENV_FILE") or Path(__file__).resolve().parents[2] / ".env")
SETTINGS = f"{HUB}/settings/tokens"


def die(message: str) -> None:
    sys.exit(f"check_hf_token: {message}")


def hf_token() -> str:
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key == "HF_TOKEN" and value.strip().strip("\"'"):
                return value.strip().strip("\"'")
    return os.environ.get("HF_TOKEN", "")


def hub_get(path: str, token: str) -> tuple[int, str]:
    request = urllib.request.Request(HUB + path, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("X-Error-Message") or error.reason


def main() -> None:
    token = hf_token()
    if not token:
        die(f"HF_TOKEN is not set in {ENV_FILE} or the environment")
    status, body = hub_get("/api/whoami-v2", token)
    if status != 200:
        die(f"the Hub rejected HF_TOKEN ({status}: {body}); create a read-only token at {SETTINGS}")
    who = json.loads(body)
    access = who["auth"].get("accessToken", {})
    role = access.get("role")
    permissions = list(access.get("fineGrained", {}).get("global", [])) + [
        p for scope in access.get("fineGrained", {}).get("scoped", []) for p in scope["permissions"]]
    if role == "write" or (role == "fineGrained" and any(not p.endswith(".read") for p in permissions)) \
            or role not in {"read", "fineGrained"}:
        die(f"HF_TOKEN of user {who['name']} is not read-only (role {role!r}). The agent can read the token, "
            f"so create a 'Read' token at {SETTINGS} and put it in {ENV_FILE}.")
    denied = []
    for model in sys.argv[1:]:
        status, message = hub_get(f"/api/models/{model}/auth-check", token)
        if status != 200:
            denied.append(f"  {model}: {status} {message}")
    if denied:
        die(f"HF_TOKEN of user {who['name']} cannot access:\n" + "\n".join(denied)
            + "\nAccept the model conditions on the Hub while logged in as that user.")
    print(f"hf token: user {who['name']}, role {role}, access to {', '.join(sys.argv[1:]) or 'no models'} verified")


if __name__ == "__main__":
    main()
