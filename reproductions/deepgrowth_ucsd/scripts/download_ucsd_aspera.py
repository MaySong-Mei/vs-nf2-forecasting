#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import http.cookiejar
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path, PurePosixPath


COLLECTION_URL = "https://www.cancerimagingarchive.net/collection/ucsd-vs-longitudinal/"
FASPEX_ORIGIN = "https://faspex.cancerimagingarchive.net"
CLIENT_ID = "ff9aa63a-72e1-436f-82ef-5677eb1f7aee"
PACKAGE_ID = "1285"
PACKAGE_ROOT = "UCSD-VS-Longitudinal"
EXAM_ID = re.compile(r"^VS_\d{4}_\d{2}$")
CONTEXT = re.compile(r"context=([A-Za-z0-9+/=]+)")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Download an explicit UCSD-VS-Longitudinal path list through IBM Aspera Connect"
    )
    result.add_argument("--paths", required=True, type=Path)
    result.add_argument("--destination", required=True, type=Path)
    result.add_argument("--batch-size", type=int, default=20)
    result.add_argument("--limit", type=int)
    result.add_argument("--wait", action="store_true")
    result.add_argument("--poll-seconds", type=float, default=10.0)
    result.add_argument("--timeout-seconds", type=float, default=24 * 60 * 60)
    result.add_argument(
        "--stall-seconds",
        type=float,
        default=30 * 60,
        help="fail audibly if no additional requested file completes for this long",
    )
    result.add_argument("--collection-url", default=COLLECTION_URL)
    result.add_argument(
        "--connect-uri-file",
        type=Path,
        default=Path.home()
        / "AppData"
        / "Local"
        / "Aspera"
        / "Aspera Connect"
        / "var"
        / "run"
        / "http.uri",
    )
    return result


def request_json(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    method: str = "GET",
    payload: object | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> object:
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request_headers = {"Accept": "application/json"}
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(
        url, data=body, headers=request_headers, method=method
    )
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_public_context(collection_html: str) -> tuple[str, dict[str, object]]:
    match = CONTEXT.search(collection_html)
    if not match:
        raise ValueError("official collection page does not contain a Faspex public context")
    encoded = match.group(1)
    padded = encoded + "=" * (-len(encoded) % 4)
    try:
        context = json.loads(base64.b64decode(padded).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("official Faspex public context is malformed") from error
    if context.get("resource") != "packages" or str(context.get("package_id")) != PACKAGE_ID:
        raise ValueError("official Faspex link points to an unexpected package")
    return encoded, context


def load_paths(path_file: Path, limit: int | None = None) -> list[str]:
    paths: list[str] = []
    for line_number, raw in enumerate(path_file.read_text(encoding="utf-8").splitlines(), 1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or len(path.parts) != 3:
            raise ValueError(f"unsafe or unexpected path on line {line_number}: {value!r}")
        package_root, exam_id, filename = path.parts
        if package_root != PACKAGE_ROOT or not EXAM_ID.fullmatch(exam_id):
            raise ValueError(f"unexpected UCSD path on line {line_number}: {value!r}")
        if not filename.startswith(exam_id + "_") or not filename.endswith(".nii.gz"):
            raise ValueError(f"unexpected UCSD filename on line {line_number}: {filename!r}")
        paths.append(value)
    if not paths:
        raise ValueError("path list is empty")
    if len(paths) != len(set(paths)):
        raise ValueError("path list contains duplicates")
    return paths[:limit] if limit is not None else paths


def read_connect_endpoint(uri_file: Path) -> tuple[str, str]:
    lines = uri_file.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2 or not lines[1].strip():
        raise ValueError("IBM Aspera Connect local API metadata is incomplete")
    base_url = lines[0].strip().rstrip("/")
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("IBM Aspera Connect API is not bound to localhost")
    return base_url, lines[1].strip()


def authorize_public_package(
    opener: urllib.request.OpenerDirector, collection_url: str
) -> str:
    with opener.open(collection_url, timeout=60) as response:
        collection_html = response.read().decode("utf-8", errors="replace")
    encoded_context, _ = extract_public_context(collection_html)
    redirect_uri = FASPEX_ORIGIN + "/aspera/faspex/token"
    authorize_query = urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": redirect_uri,
            "state": encoded_context,
        }
    )
    authorize_url = (
        FASPEX_ORIGIN
        + "/aspera/faspex/auth/authorize_public_link?"
        + authorize_query
    )
    try:
        with opener.open(authorize_url, timeout=60) as response:
            redirected = urllib.parse.urlparse(response.geturl())
            response.read()
    except urllib.error.URLError:
        # The URL contains the public package context (including its passcode),
        # so do not chain or print the underlying exception/URL.
        raise RuntimeError("Faspex public-link authorization request failed") from None
    query = urllib.parse.parse_qs(redirected.query)
    if redirected.path != "/aspera/faspex/token" or not query.get("code"):
        raise RuntimeError("Faspex public authorization did not return an authorization code")
    token = request_json(
        opener,
        FASPEX_ORIGIN + "/aspera/faspex/auth/token",
        method="POST",
        payload={
            "code": query["code"][0],
            "state": query.get("state", [encoded_context])[0],
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "redirect_uri": redirect_uri,
        },
    )
    if not isinstance(token, dict) or not token.get("access_token"):
        raise RuntimeError("Faspex token response did not include an access token")
    return str(token["access_token"])


def transfer_spec(
    opener: urllib.request.OpenerDirector, access_token: str, paths: list[str]
) -> dict[str, object]:
    selections = [
        {"path": "/" + path, "basename": PurePosixPath(path).name, "type": "symbolic_link"}
        for path in paths
    ]
    spec = request_json(
        opener,
        FASPEX_ORIGIN
        + f"/aspera/faspex/api/v5/packages/{PACKAGE_ID}/transfer_spec/download"
        + "?transfer_type=connect&type=received",
        method="POST",
        payload={"paths": selections},
        headers={"Authorization": access_token},
    )
    if not isinstance(spec, dict) or spec.get("direction") != "receive":
        raise RuntimeError("Faspex returned an invalid receive transfer specification")
    spec_paths = spec.get("paths")
    if not isinstance(spec_paths, list) or len(spec_paths) != len(paths):
        raise RuntimeError("Faspex transfer specification path count does not match the request")
    return spec


def start_batch(
    opener: urllib.request.OpenerDirector,
    connect_base: str,
    connect_key: str,
    app_id: str,
    destination: Path,
    relative_paths: list[str],
    spec: dict[str, object],
) -> object:
    destination.mkdir(parents=True, exist_ok=True)
    spec["destination_root"] = str(destination.resolve())
    spec["create_dir"] = True
    spec_paths = spec["paths"]
    assert isinstance(spec_paths, list)
    for item, relative in zip(spec_paths, relative_paths, strict=True):
        if not isinstance(item, dict):
            raise RuntimeError("Faspex returned a non-object transfer path")
        item["destination"] = relative
    request_id = str(uuid.uuid4())
    settings = {
        "allow_dialogs": False,
        "use_absolute_destination_path": True,
        "return_files": True,
        "return_paths": True,
        "app_id": app_id,
        "request_id": request_id,
        "back_link": COLLECTION_URL,
    }
    item = {
        "transfer_spec": spec,
        "aspera_connect_settings": settings,
        "authorization_key": connect_key,
    }
    return request_json(
        opener,
        connect_base + "/v5/connect/transfers/start",
        method="POST",
        payload={
            "transfer_specs": [item],
            "authorization_key": connect_key,
            "aspera_connect_settings": {"app_id": app_id},
        },
        headers={
            "Origin": FASPEX_ORIGIN,
            "Referer": FASPEX_ORIGIN + "/aspera/faspex/",
        },
    )


def completed_paths(destination: Path, paths: list[str]) -> int:
    return sum(
        1
        for relative in paths
        if (destination / Path(*PurePosixPath(relative).parts)).is_file()
        and (destination / Path(*PurePosixPath(relative).parts)).stat().st_size > 0
    )


def transfer_activity(
    opener: urllib.request.OpenerDirector,
    connect_base: str,
    connect_key: str,
    app_id: str,
    iteration_token: int,
) -> tuple[int, list[dict[str, object]]]:
    activity = request_json(
        opener,
        connect_base + "/v5/connect/transfers/activity",
        method="POST",
        payload={
            "iteration_token": iteration_token,
            "authorization_key": connect_key,
            "aspera_connect_settings": {"app_id": app_id},
        },
        headers={
            "Origin": FASPEX_ORIGIN,
            "Referer": FASPEX_ORIGIN + "/aspera/faspex/",
        },
    )
    if not isinstance(activity, dict):
        raise RuntimeError("Connect returned invalid transfer activity")
    transfers = activity.get("transfers", [])
    if not isinstance(transfers, list) or any(
        not isinstance(item, dict) for item in transfers
    ):
        raise RuntimeError("Connect returned invalid transfer activity entries")
    return int(activity.get("iteration_token", iteration_token)), transfers


def wait_for_files(
    destination: Path,
    paths: list[str],
    poll_seconds: float,
    timeout_seconds: float,
    stall_seconds: float,
    local_opener: urllib.request.OpenerDirector,
    connect_base: str,
    connect_key: str,
    app_id: str,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_progress = time.monotonic()
    last_completed = -1
    iteration_token = 0
    while True:
        completed = completed_paths(destination, paths)
        if completed != last_completed:
            print(f"Download progress: {completed}/{len(paths)} non-empty files", flush=True)
            last_completed = completed
            last_progress = time.monotonic()
        if completed == len(paths):
            return
        iteration_token, transfers = transfer_activity(
            local_opener, connect_base, connect_key, app_id, iteration_token
        )
        failed = [
            str(item.get("status"))
            for item in transfers
            if str(item.get("status", "")).lower()
            in {"failed", "cancelled", "canceled", "removed"}
        ]
        if failed:
            raise RuntimeError(
                f"Connect reported {len(failed)} failed/cancelled transfer(s): {failed}; "
                "rerun the same command to resubmit missing files"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"download timed out with {completed}/{len(paths)} files complete; "
                "rerun the same command to resume missing files"
            )
        if time.monotonic() - last_progress >= stall_seconds:
            raise TimeoutError(
                f"download stalled with {completed}/{len(paths)} files complete; "
                "rerun with a smaller --batch-size to resubmit only missing files"
            )
        time.sleep(poll_seconds)


def main() -> int:
    args = parser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.stall_seconds <= 0:
        raise SystemExit("--stall-seconds must be positive")
    paths = load_paths(args.paths, args.limit)
    destination = args.destination.expanduser().resolve()
    missing = [
        path
        for path in paths
        if not (destination / Path(*PurePosixPath(path).parts)).is_file()
        or (destination / Path(*PurePosixPath(path).parts)).stat().st_size == 0
    ]
    if not missing:
        print(f"All {len(paths)} requested files are already present and non-empty.")
        return 0

    cookie_jar = http.cookiejar.CookieJar()
    remote_opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    local_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    access_token = authorize_public_package(remote_opener, args.collection_url)
    connect_base, connect_key = read_connect_endpoint(args.connect_uri_file)
    app_id = base64.b64encode(uuid.uuid4().bytes).decode("ascii")
    print(
        f"Submitting {len(missing)} missing files in "
        f"{(len(missing) + args.batch_size - 1) // args.batch_size} batch(es)."
    )
    for offset in range(0, len(missing), args.batch_size):
        batch = missing[offset : offset + args.batch_size]
        spec = transfer_spec(remote_opener, access_token, batch)
        start_batch(
            local_opener,
            connect_base,
            connect_key,
            app_id,
            destination,
            batch,
            spec,
        )
        print(
            f"Submitted batch {offset // args.batch_size + 1}: "
            f"{offset + 1}-{offset + len(batch)}",
            flush=True,
        )
    if args.wait:
        wait_for_files(
            destination,
            paths,
            args.poll_seconds,
            args.timeout_seconds,
            args.stall_seconds,
            local_opener,
            connect_base,
            connect_key,
            app_id,
        )
        print(f"Download complete: {len(paths)} files in {destination}")
    else:
        print("Transfers submitted. Rerun with --wait to monitor filesystem completion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
