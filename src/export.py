"""
Export Morse scattering model in MLIR (StableHLO)
"""
import os

import jax
import jax.numpy as jnp
from morse_wavelet import morse_scatter, ScatteringConfig

RELATIVE_OUTPUT_FILE = "../output/morse_scattering.mlir"

def exporting_configuration(cfg: ScatteringConfig):
    scatter, meta = morse_scatter(cfg)
    x_spec = jax.ShapeDtypeStruct((cfg.B, cfg.N), jnp.float32)
    exported = jax.export.export(scatter)(x_spec)
    return scatter, meta, exported

def main():
    # Setting up the export
    cfg = ScatteringConfig()
    scatter, meta, exported = exporting_configuration(cfg)

    # Printing metadata
    print(f"J={cfg.J} Q1={cfg.Q1} Q2={cfg.Q2} T={cfg.T}  "
          f"P0={meta['P0']} P1={meta['P1']} P2={meta['P2']} dimension={meta['dimension']}")
    print(f"in_avals:  {exported.in_avals}")
    print(f"out_avals: {exported.out_avals}")
    print(f"platforms: {exported.platforms}")

    # Export to MLIR (StableHLO)
    os.makedirs(os.path.dirname(RELATIVE_OUTPUT_FILE), exist_ok=True)

    mlir_text = exported.mlir_module()
    with open(RELATIVE_OUTPUT_FILE, "w") as f:
        f.write(mlir_text)
    print(f"wrote {RELATIVE_OUTPUT_FILE}  ({len(mlir_text):,} chars)")



if __name__ == "__main__":
    main()
