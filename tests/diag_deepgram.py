"""Why does the Deepgram upload fail?  Usage: python -m tests.diag_deepgram [audio_file]
Step 1-2: DNS/TLS + auth.  Step 3: upload speed with a dummy payload.  Step 4: your file's converted size."""
import os, socket, sys, tempfile, time
from pathlib import Path
import requests
from dotenv import load_dotenv
load_dotenv()
key = os.getenv("DEEPGRAM_API_KEY")
H = {"Authorization": f"Token {key}"}
print("1. DNS:", end=" ")
try: print(socket.gethostbyname("api.deepgram.com"))
except Exception as e: print("FAIL", e); sys.exit(1)
print("2. Auth (GET /v1/projects):", end=" ")
try: r = requests.get("https://api.deepgram.com/v1/projects", headers=H, timeout=15); print(r.status_code, "(200 = key OK, 401/403 = bad key)")
except Exception as e: print("FAIL", type(e).__name__, e); sys.exit(1)
for mb in (1, 5, 20):
    print(f"3. Upload {mb} MB of zeros ->", end=" ", flush=True)
    t = time.time()
    try:
        r = requests.post("https://api.deepgram.com/v1/listen?model=nova-3", headers={**H, "Content-Type": "audio/flac"},
                          data=b"\0" * (mb << 20), timeout=(10, 120))
        print(f"HTTP {r.status_code} in {time.time()-t:.1f}s (400 is expected: zeros aren't audio; it means the upload itself worked)")
    except Exception as e:
        print(f"FAIL after {time.time()-t:.1f}s: {type(e).__name__}: {e}"); break
if len(sys.argv) > 1:
    from common.audio import to_16k_mono
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "a.flac"; to_16k_mono(Path(sys.argv[1]), f, "flac")
        print(f"4. Your file as 16 kHz mono FLAC: {f.stat().st_size/1e6:.1f} MB")
