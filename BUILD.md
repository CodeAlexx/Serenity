# serenity-trainer — build & reuse wiring

## Reuse-mechanism decision (PORT_MAP §7 #1)
**Cross-repo `-I` import, reusing mojodiffusion — NO vendoring.**
The new code lives in `/home/alex/serenity-trainer/src/serenity_trainer/`. The
numerical foundation (`serenitymojo/{autograd,tensor,ops}`) is imported from
`/home/alex/mojodiffusion` via include paths. No copy of those files is made;
mojodiffusion stays the single source for the tape/ops.

Fallback (only if cross-repo `-I` proves brittle): vendor a read-only snapshot of
`serenitymojo/{autograd.mojo,tensor.mojo,ops/}` into `serenity-trainer/`. Not done
unless T0 forces it.

## Toolchain
Mojo 1.0.0b1 via pixi (MAX 26.3). For now we reuse mojodiffusion's already-installed
pixi env (avoids a second multi-GB `pixi install`); `serenity-trainer/pixi.toml` exists
so the project can become standalone later with its own `pixi install`.

## Run the T0 gate (needs a free GPU — JIT runs a kernel)
```bash
cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
  timeout 180 prlimit --as=24000000000 \
    pixi run mojo run -I . -I /home/alex/serenity-trainer/src \
    /home/alex/serenity-trainer/smoke/t0_reuse_smoke.mojo
```
Use the `timeout`/`prlimit` cap for Mojo commands launched from VSCode. Broad
Mojo compiles have reached 61-62 GiB RSS on this machine and can trigger a
kernel OOM kill inside the VSCode snap scope if run uncapped.

Expected output:
```
T0 OK — reuse wiring + bf16 tape backward verified
  pred dtype = BF16  shape = 4 x 2
  grad(a) dtype = BF16  n = 12  finite = True
  grad(b) dtype = BF16  n = 6  finite = True
```
This proves: (1) cross-repo reuse import resolves, (2) the reused tape runs
matmul→MSE→backward, (3) BF16 in → BF16 grads out (dtype policy holds).

## Status
- T0 scaffold: WRITTEN (pixi.toml, src/serenity_trainer/__init__.mojo, smoke/t0_reuse_smoke.mojo).
- T0 run-gate: **DEFERRED — GPU busy (2026-06-05).** Run the command above when free.
- Until T0 passes, no T1+ (bf16 tape confirm / pipeline / model) code is written.
