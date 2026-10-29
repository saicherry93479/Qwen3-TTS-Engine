"""Open-loop (Poisson arrivals) load harness for the Qwen3-TTS engine: time to first audio and real-time factor.

ENGINE INTERFACE (what the harness needs; implement it in engine/engine.py and expose `make_engine(args) -> engine`):

    eng.sample_rate            int, PCM sample rate (24000)
    eng.add_request(text, lang, voice=None, seed=None, max_frames=None) -> int | False
                               THREAD-SAFE (called from the arrival thread).  Returns a request id, or False when the
                               request is refused (admission cap).  Must return immediately (enqueue only).
    eng.step() -> bool         one engine iteration (admit arrivals, one frame for every live stream, decode what is due).
                               Returns False when there was nothing to do.  Called only from the serve thread.
    eng.pop_events() -> list   called from the serve thread right after step().  Each event is a tuple
                               (rid, kind, t, n_samples):  kind "audio" = n_samples of PCM became AVAILABLE to send at
                               time t (time.perf_counter() clock; the first "audio" event of a request is its first
                               audio), kind "done" = request finished at time t (n_samples = 0).
                               If PCM is produced in another process, t must be taken there with CLOCK_MONOTONIC
                               (time.perf_counter() is CLOCK_MONOTONIC on Linux, so the values are comparable).
    eng.live() -> int          streams admitted and not finished.
    eng.stats() -> dict        cumulative counters; the harness uses "steps", "rejected" and, when present, the GPU split
                               "gpu_n" (number of timed steps) with "gpu_talker_ms", "gpu_cp_ms", "gpu_dec_ms",
                               "gpu_prefill_ms" (sums).  Anything else is passed through to the JSON.
    eng.close()

Definitions used in every table:
    offered streams N  = arrival rate x mean utterance seconds (rate = N / mean_utt_s; mean_utt_s from the stock clips).
                         N offered is NOT N live: an unpaced engine finishes early, live ~= N x RTF.
    TTFA               = first "audio" event time - add_request time (includes queueing and admission).
    RTF                = (done time - add_request time) / audio seconds of that request.
    measured           = requests that arrived inside the window (after warm-up); rejected ones are counted separately.

Usage:
    python engine/bench_open.py --engine fake --streams-list 64,128 --duration 20 --warmup 5       # harness self-test
    python engine/bench_open.py --model 1.7B --burst-list 64,128,256 --streams-list 64,128,200,256,300 --duration 300 --warmup 45 --json-out results/open_1.7B.json
"""
import os, sys, time, threading, random, argparse, json, heapq
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from eval.eval_texts_q3 import ALL, EN, LANG_NAME

FRAME_S = 0.08


def pct(a, q): return float(np.percentile(a, q)) if len(a) else float("nan")


def load_frames_per_text():
    """Mean frames per gate sentence from the stock runs (used only to convert 'offered streams' into an arrival rate)."""
    acc = {}
    sdir = f"{ROOT}/samples"
    for d in sorted(os.listdir(sdir)) if os.path.isdir(sdir) else []:
        p = f"{sdir}/{d}/meta.json"
        if d.startswith("stock_") and os.path.exists(p):
            for k, v in json.load(open(p))["clips"].items(): acc.setdefault(k, []).append(v["frames"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


class FakeEngine:
    """Sleeps instead of computing: step = a + b * live ms, every request needs its estimated frames, first audio after
    `first_frames` frames.  Only for testing the harness."""
    sample_rate = 24000

    def __init__(self, a_ms=6.0, b_ms=0.08, first_frames=2, chunk_frames=4, max_live=None, prefill_ms=3.0):
        self.a, self.b, self.first, self.chunk, self.max_live, self.prefill = a_ms, b_ms, first_frames, chunk_frames, max_live, prefill_ms
        self.lock = threading.Lock(); self.pending = []; self.act = {}; self.ev = []; self.n = 0
        self.st = {"steps": 0, "rejected": 0, "gpu_n": 0, "gpu_talker_ms": 0.0, "gpu_cp_ms": 0.0, "gpu_dec_ms": 0.0, "gpu_prefill_ms": 0.0}

    def add_request(self, text, lang, voice=None, seed=None, max_frames=None):
        with self.lock:
            if self.max_live is not None and len(self.act) + len(self.pending) >= self.max_live: self.st["rejected"] += 1; return False
            self.n += 1; need = max(8, int(len(text) * 0.62)) if max_frames is None else max_frames      # ~0.62 frames per character
            self.pending.append((self.n, need)); return self.n

    def step(self):
        with self.lock: new, self.pending = self.pending, []
        for rid, need in new: self.act[rid] = [need, 0, 0]                                              # need, done, emitted
        if not self.act: return False
        ms = self.a + self.b * len(self.act) + self.prefill * len(new); time.sleep(ms / 1e3); now = time.perf_counter()
        self.st["steps"] += 1; self.st["gpu_n"] += 1; self.st["gpu_talker_ms"] += 0.3 * ms; self.st["gpu_cp_ms"] += 0.5 * ms; self.st["gpu_dec_ms"] += 0.2 * ms
        self.st["gpu_prefill_ms"] += self.prefill * len(new)
        for rid in list(self.act):
            s = self.act[rid]; s[1] += 1
            due = self.first if s[2] == 0 else self.chunk
            if s[1] - s[2] >= due or s[1] >= s[0]:
                self.ev.append((rid, "audio", now, (s[1] - s[2]) * 1920)); s[2] = s[1]
            if s[1] >= s[0]: self.ev.append((rid, "done", now, 0)); del self.act[rid]
        return True

    def pop_events(self): ev, self.ev = self.ev, []; return ev
    def live(self): return len(self.act)
    def stats(self): return dict(self.st)
    def close(self): pass


def build_engine(a):
    if a.engine == "fake": return FakeEngine(max_live=a.max_live)
    from engine.engine import make_engine                      # implemented by the ENGINE integration
    return make_engine(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="real", choices=["real", "fake"]); ap.add_argument("--model", default="1.7B")
    ap.add_argument("--streams-list", default=None, help="open-loop offered loads in ONE boot, e.g. 64,128,200,256,300")
    ap.add_argument("--burst-list", default=None, help="hold N live streams with no arrivals: ms per step and GPU split")
    ap.add_argument("--duration", type=float, default=90); ap.add_argument("--warmup", type=float, default=15)
    ap.add_argument("--max-live", type=int, default=None); ap.add_argument("--seats", type=int, default=512)
    ap.add_argument("--sentences", type=int, default=6, help="gate sentences concatenated per request (6 English ~ 28 s)")
    ap.add_argument("--mix", default="en", choices=["en", "all"], help="'all' = single sentences in all gate languages")
    ap.add_argument("--burst-frames", type=int, default=600); ap.add_argument("--burst-window", type=float, default=10.0)
    ap.add_argument("--seeded", action="store_true"); ap.add_argument("--json-out", default=None)
    ap.add_argument("--extra", default="", help="free-form options passed to make_engine through args.extra")
    a = ap.parse_args()

    fpt = load_frames_per_text(); default_fr = float(np.mean(list(fpt.values()))) if fpt else 58.0
    if a.mix == "all" or a.sentences <= 1: pool = [(LANG_NAME[l], t, fpt.get(k, default_fr)) for l, k, t in ALL if a.mix == "all" or l == "en"]
    else:
        rng0 = random.Random(7); keys = [(k, t) for l, k, t in ALL if l == "en"]; pool = []
        for _ in range(64):
            pick = rng0.sample(keys, a.sentences); pool.append(("English", " ".join(t for _, t in pick), sum(fpt.get(k, default_fr) for k, _ in pick)))
    est_s = float(np.mean([p[2] for p in pool])) * FRAME_S
    print(f"[bench] request pool {len(pool)} texts, estimated mean {est_s / FRAME_S:.0f} frames = {est_s:.1f} s per request", flush=True)

    t0 = time.time(); eng = build_engine(a); print(f"[bench] engine boot {time.time() - t0:.0f} s", flush=True)
    sr = eng.sample_rate; lock = threading.Lock(); recs = {}; stop = threading.Event(); crashed = []

    def serve():
        try:
            while not stop.is_set():
                did = eng.step()
                for rid, kind, t, n in eng.pop_events():
                    with lock:
                        r = recs.get(rid)
                        if r is None: continue
                        if kind == "audio":
                            if r["t_first"] is None: r["t_first"] = t
                            r["samples"] += n
                        elif kind == "done": r["t_end"] = t
                if not did: time.sleep(0.0005)
        except Exception as e:                                  # noqa
            import traceback; traceback.print_exc(); crashed.append(repr(e))
    th = threading.Thread(target=serve, daemon=True); th.start()

    if os.environ.get("Q3_WATCHDOG") == "1":
        import faulthandler; faulthandler.dump_traceback_later(60, repeat=True, file=sys.stderr)
        def wd():
            while not stop.is_set():
                time.sleep(10); st = eng.stats(); print(f"[wd] steps={st.get('steps')} live={eng.live()} rejected={st.get('rejected')} serve_alive={th.is_alive()}", flush=True)
        threading.Thread(target=wd, daemon=True).start()

    def submit(text, lang, measured, seed=None, max_frames=None):
        now = time.perf_counter()
        with lock:                                              # hold the lock so the serve thread cannot see the id before the record exists
            rid = eng.add_request(text, lang, seed=seed, max_frames=max_frames)
            if rid is False: return None
            recs[rid] = {"t_sub": now, "t_first": None, "t_end": None, "samples": 0, "measured": measured}
        return rid

    def wait_done(ids, timeout):
        t_dead = time.perf_counter() + timeout
        while time.perf_counter() < t_dead and not crashed:
            with lock:
                if all(recs[i]["t_end"] is not None for i in ids): return True
            time.sleep(0.02)
        return False

    def gpu_split(s0, s1):
        gn = max(1, s1.get("gpu_n", 0) - s0.get("gpu_n", 0))
        return {k: (s1.get(k, 0.0) - s0.get(k, 0.0)) / gn for k in ("gpu_talker_ms", "gpu_cp_ms", "gpu_dec_ms", "gpu_prefill_ms")}

    w = submit(pool[0][1], pool[0][0], False, seed=1)           # warm-up request
    if w is None or not wait_done([w], 300): print("[bench] ERROR: warm-up request did not finish", flush=True); stop.set(); eng.close(); os._exit(1)
    with lock: r = recs[w]
    print(f"[bench] warm-up request: first audio {(r['t_first'] - r['t_sub']) * 1e3:.0f} ms, {r['samples'] / sr:.1f} s audio, RTF {(r['t_end'] - r['t_sub']) / (r['samples'] / sr):.3f}", flush=True)
    result = {"model": a.model, "engine": a.engine, "est_request_s": est_s, "burst": [], "loads": []}

    if a.burst_list:
        print("[burst] N streams held (fixed length, no arrivals during the window); frame budget 80 ms, RTF 0.5 = 40 ms", flush=True)
        for N in [int(x) for x in a.burst_list.split(",")]:
            ids = [submit(pool[i % len(pool)][1], pool[i % len(pool)][0], False, max_frames=a.burst_frames) for i in range(N)]; ids = [i for i in ids if i is not None]
            t_w = time.perf_counter()
            while eng.live() < len(ids) and time.perf_counter() - t_w < 30: time.sleep(0.02)
            time.sleep(1.0)
            s0 = eng.stats(); t0 = time.perf_counter(); live0 = eng.live(); time.sleep(a.burst_window); s1 = eng.stats(); t1 = time.perf_counter(); live1 = eng.live()
            st = max(1, s1["steps"] - s0["steps"]); g = gpu_split(s0, s1); ms = 1000 * (t1 - t0) / st
            row = {"N": N, "admitted": len(ids), "live_start": live0, "live_end": live1, "ms_per_step": ms, "rtf_equiv": ms / 80.0, **g}; result["burst"].append(row)
            print(f"[burst] N={N:4d} live {live0}->{live1} | {ms:6.1f} ms/step (RTF {ms / 80:.2f}) | gpu talker {g['gpu_talker_ms']:5.1f} code-predictor {g['gpu_cp_ms']:5.1f} decoder {g['gpu_dec_ms']:5.1f}", flush=True)
            wait_done(ids, 300); time.sleep(1.0)

    if a.streams_list:
        for S in [int(x) for x in a.streams_list.split(",")]:
            rps = S / est_s; rng = random.Random(S); tag_ids = []; n = 0; rej = 0
            t_start = time.perf_counter(); t_meas = t_start + a.warmup; t_stop = t_meas + a.duration; s0 = None; t_next = t_start
            while True:
                now = time.perf_counter()
                if now >= t_stop or crashed: break
                if s0 is None and now >= t_meas: s0 = eng.stats(); tm0 = now
                if now < t_next: time.sleep(min(t_next - now, 0.002)); continue      # absolute schedule: a slow add_request does not thin the arrivals
                lang, text, _ = pool[rng.randrange(len(pool))]; n += 1
                rid = submit(text, lang, now >= t_meas, seed=(1000 + n) if a.seeded else None)
                if rid is None: rej += (now >= t_meas)
                else: tag_ids.append(rid)
                t_next += rng.expovariate(rps)
            s1 = eng.stats(); tm1 = time.perf_counter(); live_end = eng.live()
            wait_done(tag_ids, 180)
            with lock: rs = [recs[i] for i in tag_ids]
            m = [r for r in rs if r["measured"] and r["t_end"] is not None and r["samples"] > 0]; unf = sum(1 for r in rs if r["t_end"] is None)
            ttfa = np.array([(r["t_first"] - r["t_sub"]) * 1e3 for r in m]); rtf = np.array([(r["t_end"] - r["t_sub"]) / (r["samples"] / sr) for r in m])
            secs = np.array([r["samples"] / sr for r in m]); st = max(1, s1["steps"] - (s0 or s1)["steps"]); g = gpu_split(s0 or s1, s1)
            res = {"streams": S, "rps": rps, "started": n, "measured": len(m), "rejected": rej, "unfinished": unf,
                   "ttfa_p50": pct(ttfa, 50), "ttfa_p95": pct(ttfa, 95), "ttfa_p99": pct(ttfa, 99), "rtf_p50": pct(rtf, 50), "rtf_p95": pct(rtf, 95), "rtf_p99": pct(rtf, 99),
                   "mean_request_s": float(secs.mean()) if len(secs) else float("nan"), "mean_request_frames": float(secs.mean() / FRAME_S) if len(secs) else float("nan"),
                   "offered_effective": rps * float(secs.mean()) if len(secs) else float("nan"), "ms_per_step": 1000 * (tm1 - tm0) / st if s0 else float("nan"), "live_end": live_end, **g}
            result["loads"].append(res)
            print(f"[load] {S:4d} streams offered ({rps:5.2f} req/s, {res['mean_request_s']:4.1f} s/request -> {res['offered_effective']:5.0f} effective) | TTFA p50 {res['ttfa_p50']:6.1f} p95 {res['ttfa_p95']:6.1f} p99 {res['ttfa_p99']:6.1f} ms"
                  f" | RTF p50 {res['rtf_p50']:.3f} p95 {res['rtf_p95']:.3f} p99 {res['rtf_p99']:.3f} | step {res['ms_per_step']:5.1f} ms | started {n} measured {len(m)} rejected {rej} unfinished {unf}", flush=True)
            if crashed: break
            time.sleep(2.0)

    result["stats"] = {k: v for k, v in eng.stats().items() if isinstance(v, (int, float, str))}; result["crashed"] = crashed
    if a.json_out: json.dump(result, open(a.json_out, "w"), indent=1)
    print("BENCH_DONE" + (" (ENGINE CRASHED)" if crashed else ""), flush=True)
    stop.set(); time.sleep(0.05); eng.close(); os._exit(1 if crashed else 0)


if __name__ == "__main__": main()
