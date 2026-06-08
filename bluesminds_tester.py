#!/usr/bin/env python3
"""
BluesMinds Model Tester
Kiểm tra tất cả model có sẵn qua BluesMinds API Gateway
Dựa trên tài liệu: https://api.bluesminds.com/v1
"""

import os
import sys
import json
import time
import csv
import getpass
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Màu sắc cho terminal
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    class Fore:
        GREEN = RED = YELLOW = CYAN = MAGENTA = RESET = ''
    class Style:
        BRIGHT = RESET_ALL = ''

def cprint(text: str, color: str = Fore.RESET, bold: bool = False):
    """In màu nếu có colorama."""
    style = Style.BRIGHT if bold else ''
    if HAS_COLOR:
        print(f"{style}{color}{text}{Style.RESET_ALL}")
    else:
        print(text)

def get_api_key() -> str:
    """Lấy API key từ env hoặc nhập từ bàn phím."""
    # Thử đọc từ biến môi trường
    env_key = os.environ.get("BLUESMINDS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if env_key:
        use_env = input(f"🔑 Tìm thấy API key trong env: {env_key[:8]}... (y/n)? ").strip().lower()
        if use_env in ('', 'y', 'yes'):
            return env_key
    # Nếu không, yêu cầu nhập
    key = getpass.getpass("🔑 Nhập BluesMinds API key (sk-...): ").strip()
    if not key:
        cprint("❌ API key không được để trống!", Fore.RED)
        sys.exit(1)
    return key

def get_base_url() -> str:
    """Base URL mặc định cho BluesMinds."""
    default = "https://api.bluesminds.com/v1"
    url = input(f"🌐 Base URL (mặc định {default}): ").strip()
    return url if url else default

def get_concurrency() -> int:
    """Số luồng kiểm tra đồng thời."""
    while True:
        try:
            val = input("⚙️  Số luồng chạy song song (mặc định 5): ").strip()
            if not val:
                return 5
            val = int(val)
            if val > 0:
                return val
            cprint("Vui lòng nhập số nguyên dương.", Fore.YELLOW)
        except ValueError:
            cprint("Số không hợp lệ.", Fore.YELLOW)

def get_timeout() -> int:
    """Timeout cho mỗi request."""
    while True:
        try:
            val = input("⏱️  Timeout (giây, mặc định 30): ").strip()
            if not val:
                return 30
            val = int(val)
            if val > 0:
                return val
            cprint("Số dương.", Fore.YELLOW)
        except ValueError:
            pass

def get_retries() -> int:
    """Số lần retry khi gặp lỗi tạm thời (429, 5xx)."""
    while True:
        try:
            val = input("🔄 Số lần thử lại khi lỗi (mặc định 2): ").strip()
            if not val:
                return 2
            val = int(val)
            if val >= 0:
                return val
            cprint("Số không âm.", Fore.YELLOW)
        except ValueError:
            pass

def fetch_models(base_url: str, api_key: str, timeout: int) -> Optional[List[str]]:
    """
    Gọi GET /v1/models để lấy danh sách model ID.
    Trả về list các model ID hoặc None nếu lỗi.
    """
    url = f"{base_url}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        cprint(f"🔍 Đang lấy danh sách model từ {url} ...", Fore.CYAN)
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # Định dạng OpenAI: {"object":"list","data":[{"id":"gpt-4o",...}]}
        if "data" in data and isinstance(data["data"], list):
            models = [item["id"] for item in data["data"] if "id" in item]
            if models:
                cprint(f"✅ Tìm thấy {len(models)} model.", Fore.GREEN)
                return models
        cprint("⚠️  Không tìm thấy model nào trong response.", Fore.YELLOW)
        return None
    except requests.exceptions.RequestException as e:
        cprint(f"❌ Lỗi khi lấy danh sách model: {e}", Fore.RED)
        if hasattr(e, 'response') and e.response is not None:
            cprint(f"   Response: {e.response.text[:200]}", Fore.RED)
        return None

def test_model(
    base_url: str,
    api_key: str,
    model: str,
    message: str,
    timeout: int,
    max_retries: int
) -> Tuple[str, bool, float, str]:
    """
    Kiểm tra một model bằng cách gửi chat completion.
    Trả về (model, thành_công, thời_gian, thông_báo).
    """
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "max_tokens": 20,
        "temperature": 0.0
    }

    retry_count = 0
    backoff = 1  # giây

    while retry_count <= max_retries:
        start = time.time()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            elapsed = time.time() - start

            if resp.status_code == 200:
                data = resp.json()
                # Lấy nội dung phản hồi để hiển thị
                reply = ""
                try:
                    reply = data["choices"][0]["message"]["content"]
                    if len(reply) > 60:
                        reply = reply[:60] + "..."
                except (KeyError, IndexError):
                    reply = "(no content)"
                return (model, True, elapsed, f"✅ {reply}")

            # Xử lý rate limit (429)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = int(retry_after) if retry_after and retry_after.isdigit() else backoff
                if retry_count < max_retries:
                    cprint(f"⏳ Model {model} bị rate limit, chờ {wait}s rồi thử lại...", Fore.YELLOW)
                    time.sleep(wait)
                    backoff = min(backoff * 2, 60)
                    retry_count += 1
                    continue
                else:
                    return (model, False, elapsed, f"❌ Rate limit (429) sau {max_retries+1} lần")

            # Lỗi client (4xx không phải 429) hoặc server (5xx) có thể retry
            if 500 <= resp.status_code < 600 and retry_count < max_retries:
                retry_count += 1
                cprint(f"⚠️  Model {model} lỗi {resp.status_code}, thử lại lần {retry_count}...", Fore.YELLOW)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue

            # Các lỗi khác (404, 401, 403, ...) không retry
            error_text = f"HTTP {resp.status_code}: {resp.text[:100]}"
            return (model, False, elapsed, f"❌ {error_text}")

        except requests.exceptions.Timeout:
            elapsed = time.time() - start
            if retry_count < max_retries:
                retry_count += 1
                cprint(f"⏰ Model {model} timeout, thử lại lần {retry_count}...", Fore.YELLOW)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
            else:
                return (model, False, elapsed, "❌ Timeout")
        except requests.exceptions.ConnectionError:
            elapsed = time.time() - start
            if retry_count < max_retries:
                retry_count += 1
                cprint(f"🔌 Model {model} lỗi kết nối, thử lại lần {retry_count}...", Fore.YELLOW)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
            else:
                return (model, False, elapsed, "❌ Connection error")
        except Exception as e:
            elapsed = time.time() - start
            return (model, False, elapsed, f"❌ Lỗi: {str(e)[:80]}")

    return (model, False, 0, "❌ Hết số lần thử")

def save_results_csv(results: List[Tuple], filename: str):
    """Lưu kết quả ra CSV."""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Success", "Time (s)", "Message"])
        for model, success, elapsed, msg in results:
            writer.writerow([model, "OK" if success else "FAIL", f"{elapsed:.3f}", msg])
    cprint(f"📁 Đã lưu CSV: {filename}", Fore.CYAN)

def save_results_json(results: List[Tuple], filename: str):
    """Lưu kết quả ra JSON."""
    data = []
    for model, success, elapsed, msg in results:
        data.append({
            "model": model,
            "success": success,
            "time_seconds": round(elapsed, 3),
            "message": msg
        })
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    cprint(f"📁 Đã lưu JSON: {filename}", Fore.CYAN)

def main():
    cprint("\n🚀 BluesMinds Model Tester", Fore.MAGENTA, bold=True)
    cprint("Kiểm tra tất cả model có sẵn qua gateway", Fore.CYAN)
    cprint("=" * 50, Fore.CYAN)

    # Nhập cấu hình
    api_key = get_api_key()
    base_url = get_base_url()
    concurrency = get_concurrency()
    timeout = get_timeout()
    max_retries = get_retries()

    test_message = input("💬 Nội dung tin nhắn gửi thử (mặc định 'hi'): ").strip()
    if not test_message:
        test_message = "hi"

    # Lấy danh sách model
    models = fetch_models(base_url, api_key, timeout)
    if not models:
        cprint("❌ Không thể lấy danh sách model. Thoát.", Fore.RED)
        sys.exit(1)

    cprint(f"\n📋 Tổng số model: {len(models)}", Fore.CYAN)
    # Hiển thị vài model đầu
    sample = models[:10]
    cprint(f"Ví dụ: {', '.join(sample)}{'...' if len(models) > 10 else ''}", Fore.YELLOW)

    proceed = input("\n▶️  Bắt đầu kiểm tra tất cả? (y/n, mặc định y): ").strip().lower()
    if proceed not in ('', 'y', 'yes'):
        cprint("Huỷ bỏ.", Fore.YELLOW)
        sys.exit(0)

    # Bắt đầu test
    cprint(f"\n🧪 Đang kiểm tra {len(models)} model với {concurrency} luồng song song...", Fore.CYAN)
    results = []
    start_total = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_model = {
            executor.submit(
                test_model, base_url, api_key, model, test_message, timeout, max_retries
            ): model for model in models
        }
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                res_model, success, elapsed, msg = future.result()
                results.append((res_model, success, elapsed, msg))
                if success:
                    cprint(f"✓ {res_model:<35} {elapsed:5.2f}s  {msg}", Fore.GREEN)
                else:
                    cprint(f"✗ {res_model:<35} {elapsed:5.2f}s  {msg}", Fore.RED)
            except Exception as e:
                cprint(f"⚠️  Lỗi không xác định với model {model}: {e}", Fore.RED)
                results.append((model, False, 0, f"Exception: {e}"))

    total_time = time.time() - start_total
    success_count = sum(1 for _, s, _, _ in results if s)

    cprint("\n" + "=" * 50, Fore.CYAN)
    cprint(f"📊 KẾT QUẢ:", Fore.MAGENTA, bold=True)
    cprint(f"   ✅ Thành công: {success_count}/{len(models)} model", Fore.GREEN)
    cprint(f"   ❌ Thất bại:   {len(models)-success_count}/{len(models)} model", Fore.RED)
    cprint(f"   ⏱️  Tổng thời gian: {total_time:.2f} giây", Fore.CYAN)

    # Lưu kết quả
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_csv = input(f"\n💾 Lưu kết quả ra CSV? (y/n, mặc định y): ").strip().lower()
    if save_csv in ('', 'y', 'yes'):
        csv_file = f"bluesminds_test_{timestamp}.csv"
        save_results_csv(results, csv_file)

    save_json = input("💾 Lưu kết quả ra JSON? (y/n, mặc định n): ").strip().lower()
    if save_json in ('y', 'yes'):
        json_file = f"bluesminds_test_{timestamp}.json"
        save_results_json(results, json_file)

    # Hiển thị danh sách model thất bại nếu có
    if success_count < len(models):
        show_fails = input("\n🔍 Hiển thị chi tiết các model bị lỗi? (y/n): ").strip().lower()
        if show_fails in ('y', 'yes'):
            cprint("\n❌ CÁC MODEL LỖI:", Fore.RED, bold=True)
            for model, success, elapsed, msg in results:
                if not success:
                    cprint(f"   {model} - {msg}", Fore.RED)

    # Gợi ý thêm
    cprint("\n✨ Hoàn thành. Để dùng model hoạt động, chỉ cần đặt 'model' trong request.", Fore.CYAN)

if __name__ == "__main__":
    main()