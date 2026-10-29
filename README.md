# Qwen3-TTS: a high-concurrency streaming engine on one A100

This branch contains our own inference engine for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Alibaba, 12 Hz
family, `0.6B-Base` and `1.7B-Base`, voice cloning). On one A100-80GB it serves **576 offered streams of the 0.6B model**
and **480 offered streams of the 1.7B model** with time to first audio under 100 ms at the 99th percentile and a
real-time factor of 0.47–0.62. The upstream code needs 181 ms to produce one 80 ms frame at any batch size, which is 2.3
times slower than real time for a single stream. The audio is exact: same weights, same maths, same sampling
distribution as upstream.

Work started 2026-10-04 23:55 IST; the numbers below were measured by 04:55 IST on 2026-10-05. The commands in
`scripts/` and `scripts/run_opt3b.sh` reproduce them.

**Validation runs, the numbers to quote** (open-loop Poisson arrivals, 45 s warm-up + 300 s window, requests of about
25 s of audio, graphed admission, 0 requests rejected, 0 unfinished, 0 recompile-limit warnings;
`results/long2_{0.6B,1.7B}.json`):

| model | offered streams | TTFA p50 | p95 | p99 | real-time factor p50 | p95 | p99 | engine step | requests measured |
|---|---|---|---|---|---|---|---|---|---|
| 0.6B-Base | 400 | 42.8 ms | 59.4 | 64.7 | 0.344 | 0.390 | 0.396 | 26.5 ms | 4,570 |
| 0.6B-Base | 512 | 53.8 | 71.5 | 78.2 | 0.434 | 0.470 | 0.532 | 34.4 | 5,640 |
| 0.6B-Base | 576 | 70.4 | 92.9 | 98.2 | 0.581 | 0.615 | 0.620 | 45.8 | 6,375 |
| 1.7B-Base | 256 | 34.7 | 45.8 | 50.0 | 0.253 | 0.275 | 0.281 | 19.9 | 2,878 |
| 1.7B-Base | 448 | 56.1 | 73.3 | 77.3 | 0.445 | 0.473 | 0.490 | 35.2 | 4,962 |
| 1.7B-Base | 480 | 59.8 | 78.2 | 82.5 | 0.474 | 0.504 | 0.513 | 37.8 | 5,391 |



Terms used throughout:

- **Frame**: one step of the model, 80 ms of audio (12.5 frames per second).
- **TTFA** (time to first audio): from the moment a request is submitted until its first PCM samples are available on the
  host. It includes queueing and admission.
- **Real-time factor (RTF)**: wall time to generate a request divided by the seconds of audio it contains. Below 1 the
  audio is produced faster than it plays; 0.5 means twice as fast.
- **Offered streams N**: an open-loop arrival rate, chosen as N ÷ (mean request length in seconds). So N is the number
  of seconds of audio requested per second of wall time. The load lines also print the **effective** figure, the
  audio-seconds per second the requests actually contained (e.g. "576 offered → 551 effective"). N offered is **not** N
  live: the engine is **unpaced** (it generates as fast as it can), so with an RTF below 1 streams finish early and the
  number inside the engine at once is about N × RTF.
- **Open-loop**: requests arrive on a Poisson clock regardless of how the engine is doing (closed-loop tests hide
  latency by slowing the arrivals down).

> **Status.** Targets were: 0.6B more than 400 concurrent requests, 1.7B about 256, RTF 0.5–0.8, TTFA under 100 ms.
> Both are met with margin over 300 s windows: the 0.6B up to 576 offered streams, the 1.7B up to 480 (448 and 480
> validated; 512 is 3 ms over on TTFA p99 in a 90 s window). Open items:
> - **TTFA is the binding limit above the knee.** At 0.6B/640 and 1.7B/512–576 the RTF still holds (0.65–0.82 p99) and
>   only first audio misses. A fast lane that produces the first frame on a high-priority stream is built and tested on
>   the talker side (`engine/arrivals.py`: `enable_first_frame`, `admit_fast`) but **not wired into the engine**.
> - **Decoder padding**: 32% of decoded frame-rows are padding (coarse decoder batch buckets); the decoder is about 40%
>   of the GPU work.
> - **Quality gate**: on the final (graphed) arrival path one seed per model has been scored. The 0.6B is inside the
>   stock spread on every column. For the 1.7B, English matches stock and the German cell (4 sentences) is 5.8% against
>   0–1.9% for stock: about three words, needs a second seed.
> - **Scratch-seat workaround**: padded rows advance a scratch seat's position; `engine.py` resets it before each
>   replay. The proper fix belongs in `engine/talker.py`.
> - **Ramblers**: in each sweep at least one request ran to the 688-frame cap (55 s for a ~26 s text). The stock model
>   does this too (one German sentence in the gate); the cap ends it.
> - The load test uses one voice and one request shape; the per-request `seed` argument is ignored (one engine-level
>   random stream, `seed=` option); boot takes 3–5 minutes warm.

## Contents

- [How the model works](#how-the-model-works)
- [What "exact" means here](#what-exact-means-here)
- [Engine design](#engine-design)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Run one request](#run-one-request)
- [Load test](#load-test)
- [Quality gate](#quality-gate)
- [Parity tests](#parity-tests)
- [Configuration reference](#configuration-reference)
- [Results](#results)
- [What was optimised](#what-was-optimised)
- [What did not help, or was decided against](#what-did-not-help-or-was-decided-against)
- [Pitfalls](#pitfalls)
- [Limitations](#limitations)
- [Provenance and licences](#provenance-and-licences)

## How the model works

Qwen3-TTS turns text into 24 kHz audio in three stages that run once per frame. A frame is 80 ms of audio and is
described by 16 integer codes (one from each of 16 codebooks of 2,048 entries).

1. **Talker** — a causal language model with 28 layers (hidden size 1024 in the 0.6B, 2048 in the 1.7B; 16 query heads
   over 8 key/value heads). It keeps a key/value cache of everything so far and, each frame, reads one input vector and
   produces the distribution of **code 0** of the next frame.
2. **Code predictor** — a small 5-layer transformer (hidden 1024, the same size in both models). Given the talker's last
   hidden state and code 0, it produces **codes 1 to 15** one after another: 15 autoregressive steps per frame, each with
   its own embedding table and output head. Its sequence grows from 2 to 16 positions and is thrown away after the frame.
3. **Decoder** — the codec's causal decoder (114 M parameters): an 8-layer transformer over frames with a sliding window
   of 72 frames, followed by transposed and dilated convolutions that upsample by 1,920 to 24 kHz. It has no lookahead,
   so audio can be emitted frame by frame.

The talker's input for the next frame is the **sum of the 16 code embeddings of the frame just produced, plus the
embedding of the next text token** (text and audio run on two tracks that are added, not interleaved). Once the text is
used up, a padding token is added instead. Generation ends when the talker samples the end-of-speech code.

**Voice cloning prompt.** Before the first frame, the talker is given ("prefilled" with) a prompt of about 78 positions
for our reference voice: 3 role tokens, 6 positions of control codes (language tag, speaker vector from an ECAPA
encoder), then the reference clip's own codes with the reference transcript and the start of the target text overlaid on
the text track. This is upstream's default cloning mode ("ICL", in-context learning, with streaming text input).

**One request, end to end** (a sentence of 5 s of speech = 62 frames):

1. Prefill: one pass of the 28 talker layers over the 78 prompt positions fills the request's key/value cache.
2. For each of the 62 frames: one talker step (1 new position against the cache) → sample code 0 (temperature 0.9,
   top-k 50, repetition penalty 1.05 over the codes this request has already produced) → 15 code-predictor steps (sample
   each of codes 1–15, temperature 0.9, top-k 50) → sum the 16 embeddings, add the next text token → next talker input.
   That is 16 sequential network calls per frame, about 1,000 for this sentence.
3. Decode: the 16 codes of each frame become 1,920 samples. The decoder must first be given the reference clip's codes
   as context (upstream decodes reference + generated together and cuts the reference off).

Upstream runs steps 2 and 3 through nested `generate()` calls in Python, which is why it takes 181 ms per frame at any
batch size: almost all of it is interpreter and kernel-launch overhead, not arithmetic.

## What "exact" means here

- **Same maths.** With both sides in fp32 and the same inputs forced, our talker and code predictor reproduce upstream's
  logits to 4.5e-7 and 6.4e-7 relative error (1.7B: 3.5e-7 and 5.3e-7), choose the same token on 100% of 705 predictor
  calls, and the prompt embeddings are bit-identical (`engine/parity.py`).
- **Rounding only in bf16.** Both upstream and we serve in bf16. Two bf16 implementations with different kernel order
  differ by rounding: talker logits 0.65–0.8%, predictor 1.0–1.6%, same top token on 96–100% of talker steps and 87–90%
  of predictor steps (near-ties flip). The same code is 1e-6 in fp32, so this is rounding, not logic.
- **Same sampling distribution, different draws.** Temperature, top-k, repetition penalty, suppressed ids, minimum length
  and end-of-speech handling follow upstream. Random numbers are drawn on the GPU inside the captured graph, so a seed
  does not reproduce upstream's clip; the quality gate compares against the stock model's own seed-to-seed spread.
- **Decoder closer to full precision than upstream's.** Upstream runs the decoder in the model's dtype, bf16, which is
  27–31 dB SNR against an fp32 decode. Ours runs in fp16 with cached state and is 47–52 dB against the same fp32
  reference, at every chunk size.
- Nothing is skipped, shortened, quantised or approximated: no step skipping, no context window on the talker, no
  re-decoding with a truncated window.

## Engine design

Code is in `engine/`. One engine thread runs `step()` in a loop; one `step()` advances every live stream by one frame.

- **Seats.** A fixed number of seats (704 for the 0.6B, 640 for the 1.7B in the validated runs). Each seat owns its
  talker key/value cache as pages of 256 positions (`seat_len` 768 by default, about 82 MB per seat), its text queue,
  its repetition-penalty mask and a decoder state slot. A request takes a free seat and releases it when its last PCM
  has reached the host.
- **One frame graph per batch bucket.** The whole frame for all live seats — talker step, sampling of code 0, the 15
  code-predictor steps with their sampling, and the construction of the next input — is one CUDA graph (a recorded
  sequence of GPU kernels replayed without Python). There is one graph per batch size "bucket" (1, 2, 4, 8, then every
  16 rows); live seats are padded up to the next bucket. The layer functions inside are `torch.compile`d with static
  shapes. The only host read per step is code 0 (to detect end of speech), through pinned memory.
- **Code predictor.** A 16-slot key/value scratch per row that lives only for the frame; each of the 15 steps processes
  one new position against it.
- **Decoder on a side stream.** The streaming decoder keeps per-seat state (a ring of the last 72 frames of keys and
  values for its transformer, the history of every causal convolution, the overlap tails of the transposed
  convolutions; 3 MiB per seat) so each new chunk costs only its own frames. It runs on a second CUDA stream so it
  overlaps the next language-model step. Chunks follow a ramp: the first packet is 1 frame, then 2, then 4 frames
  (`ramp=1/2/4`). First packets are decoded and awaited inside the step; steady chunks are collected one step later.
- **Voice priming from a snapshot.** The decoder state after decoding the reference clip is computed once per voice and
  copied into the seat at admission. Without it a cloned clip starts with noise (11 dB instead of 48 dB).
- **Admission (graphed, the default; `engine/arrivals.py` `FastAdmit`).** Up to 8 new requests per step (`admit=8`);
  beyond `max_live` requests are refused. The prompt for our voice is 78 positions whatever the text. The first 35 (3
  role tokens, 6 control positions, 26 reference-text tokens) are identical for every request of a (voice, language):
  their keys and values are computed once and copied into the seat. The other 42 positions are prefilled by a small CUDA
  graph (one per 1, 2, 4, 8 arrivals, own memory pool and capture stream). The text projection is precomputed for every
  token id, so building the prompt is a lookup. Cost per admission call: 0.14–0.25 ms of blocked engine time and 3–4 ms
  of GPU (0.6B), against 31–41 ms blocked with the original uncompiled prefill. `fast_admit=0` restores that path; a
  voice or language whose prefix length differs (for example language "auto") also falls back to it.
- **First frame.** The last prompt position is left as the seat's pending input, so frame 0 comes out of the same bulk
  graph as everyone else's frame. TTFA is therefore: wait for the running step to end + admission + one bulk frame + the
  first 1-frame decode.
- **Fast lane (built, not wired in).** `enable_first_frame()` / `admit_fast()` run the prefill and frame 0 as one small
  graph on a priority −1 CUDA stream while the bulk graph is still running: frame 0 is ready 19–26 ms after launch with
  256–400 streams live (0.6B), outputs bit-identical to a quiet run in 40 of 40 replays. Each use stretches the bulk
  step by 6–10 ms. It is the tool for the loads where only TTFA misses.

## Repository layout

| path | what it is |
|---|---|
| `engine/engine.py` | the serving loop (`Q3Engine`: `add_request`, `step`, `pop_events`, `live`, `stats`, `close`) |
| `engine/talker.py` | talker on seats, paged key/value cache, frame graphs, on-GPU sampling |
| `engine/code_predictor.py` | the 15 predictor steps as one unit |
| `engine/prompt.py` | upstream's voice-clone prompt, with the per-voice parts cached |
| `engine/decoder_stream.py` | exact streaming decoder with per-seat state (`StreamDecoder`) |
| `engine/arrivals.py` | graphed admission (shared voice prefix + prefill graphs, the default) and first-frame fast-lane graphs (not wired in) |
| `engine/model_load.py`, `engine/ops.py` | weight extraction from the upstream checkpoint, small ops |
| `engine/bench_open.py` | open-loop load harness (TTFA, RTF, steady-state "burst" points) |
| `engine/bench_step.py`, `bench_decoder.py`, `prof_cp.py`, `prof_decoder.py`, `probe_ttfa.py` | stage benchmarks and profiles |
| `engine/parity.py`, `test_decoder_stream.py`, `test_rng.py`, `test_arrivals.py`, `test_fastlane.py` | exactness tests |
| `infer.py` | one request to a wav file |
| `eval/` | quality gate: stock clips, engine clips, scoring; `eval_texts_q3.py` holds the 32 gate sentences |
| `eval/bench_stock.py` | timing of the upstream code |
| `scripts/` | `download_weights.sh`, `run_load.sh`, `run_validate.sh`, `run_gate.sh`, `run_stock_baseline.sh`, `run_opt3.sh`, `scripts/run_opt3b.sh` |

Not in the repository: `weights/`, the Python environment, the upstream checkout `Qwen3-TTS/`, generated audio.

## Setup

Tested on one A100-80GB, Linux, Python 3.11, CUDA 12.8.

```bash
git clone -b qwen3-tts <this repository> qwen3tts && cd qwen3tts
git clone https://github.com/QwenLM/Qwen3-TTS.git            # upstream code, used for weights loading and as the reference
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e ./Qwen3-TTS      # installs qwen-tts 0.1.1 and pins transformers==4.57.3
uv pip install --python .venv/bin/python flash-attn==2.8.3 "huggingface_hub[cli]" hf_transfer soundfile librosa
scripts/download_weights.sh                                  # 0.6B-Base, 1.7B-Base, tokenizer -> weights/ (about 7.4 GB)
```

Versions used: torch 2.8.0+cu128, transformers 4.57.3, flash-attn 2.8.3, triton 3.4.0, qwen-tts 0.1.1. Transformers 5 is
not supported by upstream. Never commit `weights/`.

The engine needs one voice file per model, `samples/voice_ref_<model>.pt` (speaker vector, reference codes,
transcript). `gen_stock.py` writes it from `samples/voice_ref.wav` on first use; `infer.py --ref-wav` builds one in
memory from your own clip.

## Run one request

```bash
.venv/bin/python infer.py --text "Your order will be delivered by tomorrow evening." --out out.wav
.venv/bin/python infer.py --model 1.7B --lang German --text "Guten Morgen, Ihr Termin ist bestätigt." --out de.wav
.venv/bin/python infer.py --text "..." --ref-wav my_voice.wav --ref-text "exact transcript of the clip" --out mine.wav
```

`infer.py` mirrors `gen_engine.py`: it builds a `Q3Engine`, submits one request, steps the engine until the request is
done and writes the PCM. Smoke-tested once on the A100 (0.6B, stored voice, `--fast-boot`, 64 seats): engine ready in
90 s, 4.40 s of audio, first audio after 24 ms, real-time factor 0.148, 14.9 GB of GPU memory. The `--ref-wav` path
(your own voice) has not been run; it switches to the original uncompiled prefill (`fast_admit=0`), because graphed
admission caches the prompt prefix of the stored voice at boot. The first start compiles and captures graphs
(minutes); `--fast-boot` skips the decoder compile.

From Python:

```python
from engine.engine import Q3Engine
eng = Q3Engine(model="0.6B", n_seats=64, extra="seed=0,keep_audio=1")
rid = eng.add_request("Hello, this is a test.", "English")        # returns False when the engine is full
done = False
while not done:
    eng.step()
    for r, kind, t, n in eng.pop_events():                        # kind "audio": n samples ready at time t; "done"
        done |= (r == rid and kind == "done")
pcm = eng.audio[rid]                                              # list of numpy chunks at 24 kHz (keep_audio=1)
```

## Load test

```bash
scripts/run_load.sh 0.6B                  # 704 seats: 400,512,576,640 offered (15 s warm-up + 90 s windows)
scripts/run_load.sh 1.7B 256,448,480,512  # 640 seats
scripts/run_validate.sh 0.6B 576          # 45 s warm-up + 300 s window: the numbers to quote
scripts/run_validate.sh 1.7B 480
scripts/run_stock_baseline.sh 0.6B        # timing of the upstream code (181 ms per frame)
```

Each run boots the engine first (3–5 minutes with a warm compile cache, about 10 minutes cold) and needs most of an
80 GB card: 68.9 GB for the 0.6B at 704 seats, 67.4 GB for the 1.7B at 640 seats. Set `SEATS=...` in the environment for a smaller seat count
(`SEATS=384 scripts/run_load.sh 0.6B 256,400`).

The scripts wrap `engine/bench_open.py`:

| flag | meaning |
|---|---|
| `--model 0.6B\|1.7B` | checkpoint |
| `--seats N`, `--max-live N` | number of seats; admission cap (requests beyond it are refused and counted) |
| `--streams-list 256,400` | open-loop offered loads, all in one boot |
| `--burst-list 128,256` | hold exactly N live streams with no arrivals: wall ms per step and the GPU time per stage |
| `--duration S`, `--warmup S` | measurement window and warm-up per load |
| `--sentences 6` | gate sentences joined per request (6 English sentences ≈ 26 s) |
| `--mix en\|all` | request texts: English, or single sentences in all gate languages |
| `--extra "k=v,k=v"` | engine options, see below |
| `--json-out path` | results as JSON |
| `--engine fake` | a sleeping fake engine, to test the harness itself |

Check `grep -c recompile_limit logs/<run>.log` is 0: above it, a compiled function silently runs uncompiled.

## Quality gate

32 sentences (24 English, 4 Spanish, 4 German), one reference voice, scored with Whisper-large-v3 (word error rate) and
WavLM speaker vectors (similarity). The stock model is run with two seeds to measure its own seed-to-seed variation; the
engine must land inside that spread.

```bash
scripts/run_gate.sh 0.6B 1000             # stock clips if missing, engine clips, scores
.venv/bin/python make_page.py stock_0.6B_s1000 engine_0.6B_s1000      # static listening page in samples/page/
```

Original (uncompiled) arrival path:

| run | WER all | WER English | similarity to voice reference | similarity to stock clip, same sentence | duration ratio |
|---|---|---|---|---|---|
| stock 0.6B, seed 1000 | 5.7% | 5.5% | 0.937 | reference | |
| stock 0.6B, seed 2000 | 7.3% | 5.2% | 0.930 | 0.966 | ×1.01 |
| engine 0.6B, seed 1000 | 5.9% | 5.8% | 0.933 | 0.961 | ×1.00 |
| engine 0.6B, seed 2000 | 5.7% | 5.8% | 0.935 | 0.967 | ×1.00 |
| stock 1.7B, seed 1000 | 4.9% | 5.5% | 0.930 | reference | |
| stock 1.7B, seed 2000 | 5.1% | 5.2% | 0.928 | 0.973 | ×1.04 |
| engine 1.7B, seed 1000 | 5.3% | 5.8% | 0.925 | 0.967 | ×1.01 |

Graphed admission (the default path, `results/gate_engine_{0.6B,1.7B}_fa.json`), seed 1000:

| run | WER all | WER English | WER Spanish | WER German | similarity to voice reference | similarity to stock clip | duration ratio |
|---|---|---|---|---|---|---|---|
| **engine 0.6B** | 6.1% | 6.0% | 6.7% | 5.8% | 0.932 | 0.963 | ×0.99 |
| **engine 1.7B** | 5.7% | 5.5% | 6.7% | 5.8% | 0.929 | 0.970 | ×1.02 |

Most English "errors" are the scorer writing numbers as digits; the worst sentences are the same in every run. The stock
0.6B rambled on one German sentence at seed 2000 (25% German WER), which is why its total is 7.3%. The Spanish and
German cells are 4 sentences each, so one or two words move them by several points: the 1.7B engine's German 5.8%
against 0–1.9% for the two stock seeds is about three words and has been scored on one seed only.

`scripts/run_gate.sh <model> <seed>` writes `results/gate_engine_<model>_s<seed>.json`; the files from the overnight
runs are named `gate_engine_<model>.json` (original path) and `gate_engine_<model>_fa.json` (graphed admission).

## Parity tests

```bash
DTYPE=fp32 COMPILE=off MODEL=0.6B .venv/bin/python engine/parity.py        # maths vs upstream, teacher-forced
MODEL=1.7B .venv/bin/python engine/parity.py                               # bf16, compiled (serving path)
.venv/bin/python engine/test_decoder_stream.py fp16 --graphs               # streaming decode vs one-shot fp32 decode (SNR)
.venv/bin/python engine/test_rng.py                                        # sampling inside a replayed graph draws fresh numbers
```

## Configuration reference

Engine options, passed as `--extra "k=v,k=v"` to `bench_open.py` or as `extra=` to `Q3Engine`:

| option | default | meaning |
|---|---|---|
| `fast_admit` | `1` | graphed admission with the shared voice prefix; `0` = original uncompiled prefill (also env `Q3_FAST_ADMIT`) |
| `dec_fine` | `0` | experimental: finer decoder batch buckets against the 32% padding (also env `Q3_DEC_FINE`); not validated |
| `ramp` | `1/2/4` | decoder chunk sizes in frames: first packet, second, then steady |
| `admit` | `8` | new requests admitted per step |
| `seat_len` | `768` | talker positions per seat (multiple of 256); prompt + frames must fit |
| `max_text` | `640` | text tokens kept per request |
| `dec_dtype` | `fp16` | decoder precision (`fp32` is exact to 57–61 dB and slower) |
| `dec_compile` | `default` | `off` skips `torch.compile` of the decoder (fast boot, slower decode) |
| `dec_max` | `256` | largest decoder batch bucket |
| `cap` | `100000` | frame cap per request (end of speech still ends it) |
| `seed` | `0` | seed of the engine's one on-GPU random stream (the per-request `seed` argument of `add_request` is ignored) |
| `serial` | `0` | `1` = synchronise after every stage: exact GPU split per stage, no overlap (for profiling) |
| `keep_audio` | `0` | `1` = keep the PCM per request in `eng.audio` |

Environment switches:

| variable | default | meaning |
|---|---|---|
| `QWEN3TTS_ROOT` | repository root | where `weights/`, `samples/`, `results/` are looked up |
| `Q3_TALKER_COMPILE` | `default` | `torch.compile` mode of the talker layers (`off` disables) |
| `Q3_CP_COMPILE` | `default` | same for the code predictor |
| `Q3_PF_COMPILE` | `off` | compile the layer function of the graphed prefill in `engine/arrivals.py` |
| `DEC_NHWC` | `1` | decoder convolutions as 2-D convolutions on channels-last memory |
| `DEC_RING` | `1` | decoder key/value state as a ring written in place |
| `DEC_LAZY` | `0` | experimental decoder variant from the second decoder round; not validated, leave off |
| `Q3_WATCHDOG` | unset | `1` = the load harness dumps stacks if the engine stalls |

Benchmark and test scripts read their own variables (`MODEL`, `COMPILE`, `DTYPE`, `FRAMES`, `BS`, `SEAT_LEN`, `CTX`,
`REPS`, `TAG`, `PREC`, `BUCKETS`, `CHUNKS`, `GRAPHS`, `MODE`, `TF32`, `CUDNN_BENCH`, `LAYOUT`, `PROFILE`, `B`, `N`,
`WHAT`, `LOADS`, `SEATS`, `DUR`, `WARM`, `EST_S`, `EXTRA`, `BULK`, `PF_COMPILE`, `BATCHES`); each is documented at its
point of use.

## Results

All on one A100-80GB, bf16 language model, fp16 decoder, voice-clone mode, requests of 6 English sentences (about 26 s).

**Upstream code** (`bench_stock.py`, 0.6B):

| batch | ms per frame | talker | code predictor | × real time, whole batch |
|---|---|---|---|---|
| 1 | 178 | 45.5 | 128.6 | 0.4 |
| 16 | 185 | 46.1 | 131.3 | 6.9 |
| 64 | 194 | 45.9 | 130.6 | 26.4 |

**Language-model step alone** (talker + code predictor, streams held live, no decoder, no arrivals; `results/step_*.json`):

| live streams | 0.6B, short context | 0.6B, context 420 | 1.7B, short context | 1.7B, context 420 |
|---|---|---|---|---|
| 128 | 13.4 ms | | 15.2 | 18.0 |
| 256 | 17.8 | 23.5 | 20.5 | 26.1 |
| 400 (384 for 1.7B) | 25.2 | 34.1 | 27.0 | |
| 512 | 27.9 | 39.5 | 33.2 | |
| 1024 | 48.5 | | | |

**Decoder alone** (`results/dec_cost.json`): 0.51–0.54 ms of GPU per second of audio at 4–8 frame chunks; 14.1 / 24.5 /
44.5 / 83.7 ms per call for 256 streams × 1 / 2 / 4 / 8 frames, linear in streams above 128.

**Whole engine, streams held live** (no arrivals; stage times overlap and do not add up):

| model | live | wall ms per step | RTF equivalent |
|---|---|---|---|
| 0.6B | 128 / 256 / 400 / 512 | 22.0 / 32.3 / 48.1 / 54.6 | 0.27 / 0.40 / 0.60 / 0.68 |
| 1.7B | 128 / 192 / 256 / 300 | 23.5 / 31.6 / 34.6 / 43.5 | 0.29 / 0.40 / 0.43 / 0.54 |

**Open-loop sweeps, graphed admission** (15 s warm-up + 90 s windows, so slightly optimistic; 0 unfinished everywhere;
`results/open2_0.6B.json`, `open2_1.7B.json`, `open3_1.7B.json`):

| model | offered | TTFA p50 | p95 | p99 | RTF p50 | p95 | p99 | step | rejected |
|---|---|---|---|---|---|---|---|---|---|
| 0.6B | 400 | 41.8 ms | 58.2 | 64.0 | 0.326 | 0.389 | 0.391 | 26.0 ms | 0 |
| 0.6B | 512 | 56.0 | 78.9 | 86.6 | 0.442 | 0.548 | 0.553 | 36.2 | 0 |
| 0.6B | 640 | 82.3 | 118.8 | 131.4 | 0.718 | 0.816 | 0.820 | 53.5 | 0 |
| 0.6B | 768 | 114.0 | 162.8 | 168.5 | 1.038 | 1.069 | 1.071 | 74.8 | 223 |
| 1.7B | 256 | 36.0 | 46.8 | 49.8 | 0.260 | 0.275 | 0.276 | 20.5 | 0 |
| 1.7B | 320 | 38.4 | 50.7 | 54.9 | 0.289 | 0.304 | 0.308 | 22.7 | 0 |
| 1.7B | 384 | 45.6 | 60.0 | 67.4 | 0.348 | 0.393 | 0.396 | 27.4 | 0 |
| 1.7B | 448 | 54.7 | 73.5 | 78.0 | 0.439 | 0.472 | 0.475 | 34.9 | 0 |
| 1.7B | 512 | 70.7 | 97.7 | 102.9 | 0.607 | 0.648 | 0.651 | 46.0 | 0 |
| 1.7B | 576 | 83.0 | 113.7 | 124.5 | 0.700 | 0.768 | 0.777 | 54.0 | 0 |
| 1.7B | 640 | 103.9 | 158.1 | 173.3 | 0.886 | 1.070 | 1.079 | 67.2 | 32 |

The 300 s validations are the table at the top.

**The arrival fix, before and after** (300 s windows; "before" = uncompiled prefill inside the step, `results/long_*.json`):

| model, offered | admission | TTFA p50 | p95 | p99 | RTF p50 | p99 | step |
|---|---|---|---|---|---|---|---|
| 0.6B, 400 | uncompiled | 75.3 ms | 104.7 | 109.8 | 0.539 | 0.597 | 42.3 ms |
| 0.6B, 400 | graphed | 42.8 | 59.4 | 64.7 | 0.344 | 0.396 | 26.5 |
| 1.7B, 256 | uncompiled | 63.4 | 93.1 | 99.1 | 0.355 | 0.392 | 27.8 |
| 1.7B, 256 | graphed | 34.7 | 45.8 | 50.0 | 0.253 | 0.281 | 19.9 |

Where first audio went at 0.6B / 400 offered with the uncompiled prefill (p50, `results/ttfa_breakdown_0.6B*.json`):
wait for the running step 20.8 ms, prompt build 1.7, prefill 29.9 (engine thread blocked), bulk frame 18.5, first
decode 2.5; total 74.2 ms. With graphed admission: wait 13.1, build 0.0, prefill 0.2 blocked, bulk frame 24.9, first
decode 2.5; total 41.9 ms. GPU cost per arrival: 23 ms → 4.7 ms over the sweeps.

**Stage split under load** (0.6B, 512 offered, graphed admission; stage times overlap, so they sum to more than the
36.2 ms wall step): language model 29.7 ms, decoder 21.9 ms, admission 3.9 ms per step. The language-model figure
includes contention with the overlapping decoder (the same frame takes about 25 ms alone at 400 live).

**Earlier sweeps with the uncompiled prefill** (`results/open_0.6B.json`, `open_1.7B.json`): 0.6B 256 / 320 / 400 / 448 /
512 offered gave TTFA p99 94.4 / 99.2 / 107.9 / 112.6 / 129.9 ms and RTF p50 0.34 / 0.39 / 0.52 / 0.57 / 0.70; 1.7B 128
/ 192 / 256 / 300 gave 92.7 / 93.0 / 100.6 / 103.5 ms and 0.24 / 0.26 / 0.37 / 0.41.

## What was optimised

| change | before | after | exact maths? |
|---|---|---|---|
| own frame loop instead of nested `generate()` (0.6B, 256 live, uncompiled) | 181 ms per frame | 32.6 ms | yes |
| `torch.compile` + one CUDA graph for the whole frame (0.6B, 256 live) | 32.6 ms | 26.1 ms | yes (bf16 rounding) |
| code predictor: stop the compiler copying the 16-slot key/value buffers every step (split around an in-place write; 400 live) | 35.1 ms of kernels | 19.6 ms | yes |
| → language-model step, 0.6B at 256 / 400 live | 26.1 / 42.1 ms | 17.8 / 25.2 ms | yes |
| decoder: CUDA graph only → compiled (256 streams × 4 frames) | 119.0 ms | 56.0 ms | yes |
| decoder: convolutions on channels-last memory (`DEC_NHWC`) | 56.0 ms | 45.8 ms | yes (rounding) |
| decoder: key/value ring written in place (`DEC_RING`) | 45.8 ms | 44.5 ms (1-frame calls 16.4 → 14.1 ms) | yes |
| decoder: cached per-seat state instead of re-decoding a window | not exact (5–24 dB) | 47–52 dB | this is what makes streaming exact |
| decoder compile mode `default` instead of max-autotune | ~170 s per shape | 20–45 s per shape, same speed | yes |
| admission: shared voice prefix (35 of 78 positions) + graphed prefill, prompt text as a table lookup | 31–41 ms of blocked engine time per admission call | 0.14–0.25 ms blocked, 3–4 ms of GPU | yes (fp32 1e-6; bf16 first-frame logits within 1.2%, same top token) |
| → 0.6B at 400 offered, 300 s | TTFA p99 109.8 ms, RTF 0.54, step 42.3 ms | 64.7 ms, 0.34, 26.5 ms | yes |
| → 1.7B at 256 offered, 300 s | TTFA p99 99.1 ms, RTF 0.36, step 27.8 ms | 50.0 ms, 0.25, 19.9 ms | yes |

## What did not help, or was decided against

- **Talker attention over long contexts.** The talker step roughly doubles between 80 and 420 positions of context
  (0.6B, 400 live: 6.8 → 15.7 ms). The extra time is the read of cached keys and values, 15.7 GB per step, which is 86%
  of the A100's memory bandwidth. A different attention kernel could recover about 1 ms; no work was done there.
- **Sharing a prompt prefix across requests in attention.** Only the first positions of the prompt are identical per
  voice; they are reused at prefill, not at every step.
- **bf16 decoder** (what upstream runs): 29–33 dB against fp32. Rejected in favour of fp16.
- **Re-decoding with a window of context** (25, 72 or 144 frames) and upstream's `chunked_decode(300, 25)`: 5–64 dB and
  12.8 dB against a one-shot decode; the 72-frame window applies per layer, so the true receptive field is far longer.
- **TensorRT for the decoder**: no published win at large batch, state tensors re-formatted every call, a week of work.
- **Flash-attn versus other attention backends for speed**: the workload is launch-bound, not attention-bound.
- **Finer language-model batch buckets**: already every 16 rows; padding is 0.7–3% of rows.
- Not done yet: wiring the first-frame fast lane into the engine; decoder convolutions as matrix multiplies (estimated
  floor 0.42–0.50 ms per audio-second); finer decoder batch buckets (32% of decoded frame-rows are padding today); a
  dynamic-batch decoder compile for faster boot.

## Pitfalls

1. **Sampling inside a CUDA graph needs the generator registered on the graph** (`register_generator_state`). With it,
   2,000 replays gave 1,999 distinct draws; without it every replay returns the same sample.
2. **The compiler can silently copy state buffers.** Writing the predictor's key/value scratch inside one compiled
   function made it copy the whole buffer on every step (17 of 35 ms at 400 streams). Keep in-place state writes
   outside the compiled pieces.
3. **Prime the decoder for cloned voices.** The decoder needs the reference clip's codes as state before the first
   generated frame, or the clip starts with noise (11 dB instead of 48 dB). Restore the per-voice snapshot at admission.
4. **Upstream's own chunked decode is not exact**, so "matches upstream's streaming" is not a valid test; compare against
   a one-shot fp32 decode of the same codes and measure SNR.
5. **The padded rows of a batch bucket run too.** They point at a scratch seat whose position advanced on every replay
   and ran off the end after 768 steps (a device-side assert in the first long run). Reset or freeze the scratch seat.
6. **`torch.compile` falls back to uncompiled code after 8 recompilations of one function** with only a warning. With one
   static shape per bucket the limit must be raised (done in `engine/model_load.py`); grep the log for `recompile_limit`.
7. **Each CUDA graph owner needs its own memory pool and capture stream** when graphs are replayed concurrently on
   different streams (language model, decoder, prefill).
8. **Long compiles hold the GPU.** A benchmark that compiles every bucket took 16 minutes before measuring anything; the
   compile cache (`/tmp/torchinductor_<user>` by default) persists and makes the next boot 3–4 times shorter.
9. **N offered is not N live.** Below capacity streams finish early; read the step time and the RTF, not the label.
10. **Uncompiled work on the engine thread stalls every stream.** The original prefill blocked the engine for 30 ms per
    admission; the average step at 400 offered read 42 ms although the frame itself took 18.5 ms. Measure the wall step
    and the per-stage GPU time separately, and keep anything per-arrival off the engine thread or inside a graph.
11. **A shared GPU lock needs short holds and one owner per job.** Runs were serialised with `flock gpu.lock`; one
    benchmark held it for 16 minutes of compiles while the load sweep waited, and a duplicate scoring run re-scored
    clips that were already in the results file. Check the queue and the results directory before starting a job.

## Limitations

- Ten languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian). **No Indian
  language** is supported by these checkpoints.
- The load test uses one voice, English requests of about 26 s, and one sampler seed stream; other request shapes change
  the context length and therefore the talker cost.
- Boot: 185–190 s with a warm compile cache in the validated runs (up to 5 minutes), about 10 minutes cold. GPU memory:
  68.9 GB for 704 seats of the 0.6B (key/value cache 57.8 GB), 67.4 GB for 640 seats of the 1.7B (52.6 GB), so seats
  cannot go much higher at 768 positions per seat.
- TTFA, not throughput, is the limit above 576 (0.6B) / 480 (1.7B) offered streams until the fast lane is wired in.
- Graphed admission assumes a fixed prompt-prefix length per (voice, language); other cases use the slow path.
- There is no network front end: the engine is a Python object driven by a loop.
- Upstream's streaming-text mode is kept as is, including its known behaviour that speaking rate can drift on long texts
  with long reference clips.
- Short measurement windows are optimistic; quote the 300 s runs.

## Provenance and licences

- Model, weights and upstream code: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), Apache-2.0; paper
  "Qwen3-TTS Technical Report", arXiv 2601.15621. Weights: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, `-1.7B-Base`,
  `Qwen/Qwen3-TTS-Tokenizer-12Hz` on Hugging Face.
- The engine in `engine/` is our own code and loads the upstream checkpoints unchanged.
- Attention kernels: [flash-attention](https://github.com/Dao-AILab/flash-attention) (BSD-3-Clause).
- Scoring models: `openai/whisper-large-v3`, `microsoft/wavlm-base-plus-sv`.
