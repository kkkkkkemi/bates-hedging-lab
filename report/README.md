# Validation report inputs

The reporting code reads the run CSV and JSON artifacts and regenerates the PDF, findings and figures. Use the experiment configuration associated with the output directory. The PDF, findings and figures in this folder are copied from the full run.

```text
python -m quantproj --config experiments/full.yaml --output output/full --render-only
```

The PDF includes the model use, conceptual soundness, implementation checks, calibration, hedging, attribution, commodity extension, limitations and monitoring. The source of each plotted observation is a CSV in the same run directory.
