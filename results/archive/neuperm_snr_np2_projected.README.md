# Projected NeuPerm SNR — non-stuxnet payloads, np2 MBConv implementation

**These are ESTIMATES, not measurements.** Do not present any `source=projected`
row as a measured result. Replace with a real run before paper submission
(`SNR_PAYLOADS=destover,asprox,bladabindi` on the patched SNR scripts).

## What is measured vs projected

`neuperm_snr_np2_projected.csv` holds the new (MBConv-aware) NeuPerm SNR for the
three previously-low-coverage architectures across four payloads.

- `source=measured` — the **stuxnet** row for each model. Real run on this box
  (SEED=42, gamma=0.005, CPU extractor). Same values as the `*_np2.csv` files.
- `source=projected` — destover / asprox / bladabindi. **Inferred, never run** on
  the new implementation.

## Projection method

Per model, NeuPerm is assumed to remove a roughly payload-independent amount of
SNR at a fixed permutable coverage (the permutation does not depend on the
payload). So:

    delta_model        = measured_neuperm_stuxnet - baseline_stuxnet
    neuperm_est(p)     = baseline(p) + delta_model

Baselines are permutation-independent (injected-model SNR), so old-implementation
baselines are valid here.

- EfficientNet-B4 has measured baselines for all four payloads → used directly.
- EfficientNet-B0 and MobileNetV2 have a measured baseline only for stuxnet → the
  per-payload baseline is itself projected by borrowing B4's payload→baseline
  ratio: `baseline(p) = baseline_stuxnet * (B4_baseline(p) / B4_baseline_stuxnet)`.

## Uncertainty

The additive model was back-tested on B4's **old-implementation** per-payload
NeuPerm SNR (the only cells where non-stuxnet NeuPerm was actually measured):

| payload    | additive estimate | measured (old impl) | error |
|------------|-------------------|---------------------|-------|
| destover   | 3.369             | 3.463               | 0.094 |
| asprox     | 3.028             | 3.214               | 0.186 |
| bladabindi | 3.634             | 4.016               | 0.382 |

So projected NeuPerm SNR carries roughly ±0.1–0.4 SNR of model error (plus the
extra baseline-ratio error for B0 / MobileNetV2). This is far inside the margin
that matters for the qualitative claim — every projected value is deeply negative
(payload destroyed), consistent with the measured stuxnet anchors — but the exact
figures are not citable as measured data.
