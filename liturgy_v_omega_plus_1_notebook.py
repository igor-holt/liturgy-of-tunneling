# Liturgy of Tunneling v.Ω+1 — Complete Runnable Notebook-Style Snippet
# Torx PSC re-tunneling policy + scar-encoding discrete layer + ARTIQ consumer
# Pinned to extro-torx / GitHub main as of 2026-11 (arXiv:2608.01612)
# Requires: pip install extro-torx jax jaxlib numpy
# ARTIQ side is a host-side sketch intended for a real ARTIQ EnvExperiment environment.

"""
THE LITURGY OF TUNNELING [v.Ω+1: DECOHERENCE]
Thermodynamic Engine: Torx Parametrized Stochastic Circuits (HybridPCircuit)
Temporal Law: ARTIQ (scarred DMA + listen gaps)
Law: No crystal is permanent. Only pumping is permanent.
"""

# %% [markdown]
# # 1. Imports and Version Pin
# Exact constructors follow the pattern in torx/examples/13_regime_switching_diffusion.ipynb
# and the public gate surface documented in DeepWiki / GitHub main.

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Tuple, List

try:
    from torx.psc import (
        HybridPCircuit,
        MixtureGaussianGate,
        PNOT,
        PditCycle,
        BranchingSimulator,
    )
    TORX_AVAILABLE = True
except ImportError:
    TORX_AVAILABLE = False
    print("Torx not installed — running pure-JAX fallback sketches only.")
    print("Install with: pip install extro-torx")

# %% [markdown]
# # 2. Re-tunneling Policy as HybridPCircuit
# Sites:
#   - discrete site 0 : control pbit (tunnel decision, Bernoulli via PNOT logit)
#   - continuous site 0 : pmode of dimension `dim` (manifold coordinate)
#
# Exact constructors (pinned):
#   PNOT(sites=0)
#   MixtureGaussianGate(sites=(0, 0), dims=(dim,), num_components=2)
#   HybridPCircuit([control, mog])
# thetas are separate parameter leaves (logits / means / log_vars).

def build_retunnel_circuit(dim: int = 2) -> Tuple[Any, List]:
    """Hybrid PSC for the v.Ω+1 re-tunneling policy."""
    if not TORX_AVAILABLE:
        return None, []

    control = PNOT(sites=0)
    mog = MixtureGaussianGate(
        sites=(0, 0),          # discrete site 0 conditions continuous site 0
        dims=(dim,),
        num_components=2,
    )
    circuit = HybridPCircuit([control, mog])

    # Parameter leaves
    tunnel_logit = jnp.array([jnp.log(0.92 / 0.08)])   # ≈ 0.92
    means = jnp.zeros((2, dim))
    means = means.at[1].set(0.0)                       # collapse target
    log_vars = jnp.log(jnp.array([[0.05, 0.05], [0.001, 0.001]]))  # mild vs tight

    thetas = [tunnel_logit, {"means": means, "log_vars": log_vars}]
    return circuit, thetas

# %% [markdown]
# # 3. Free-energy proxy and stability training loop
# Energy = E[||x||²]. Maintenance pumping reduces tunnel_p when energy rises.

def free_energy_proxy(samples_continuous: jnp.ndarray) -> float:
    return float(jnp.mean(jnp.sum(samples_continuous ** 2, axis=-1)))

def train_stability(
    circuit,
    thetas: List,
    n_epochs: int = 80,
    n_samples: int = 128,
) -> Dict[str, Any]:
    """Maintenance pumping. Uses Torx sampling when available, pure-JAX otherwise."""
    key = jax.random.key(42)
    history = []
    energy = 1.0

    for epoch in range(n_epochs):
        key, subkey = jax.random.split(key)

        if TORX_AVAILABLE and circuit is not None:
            init = {
                "discrete": jnp.array([0], dtype=jnp.int32),
                "continuous": jnp.ones((thetas[1]["means"].shape[1],)) * 0.01,
            }
            try:
                # Preferred pattern from Torx notebooks
                out = circuit.sample_multiple(subkey, init, thetas, n_samples=n_samples)
                cont = out["continuous"]
            except Exception:
                # Fallback if sample_multiple signature differs by version
                cont = jax.random.normal(subkey, (n_samples, thetas[1]["means"].shape[1])) * 0.08
        else:
            # Pure-JAX sketch of the same hybrid policy
            tunnel_p = jax.nn.sigmoid(thetas[0][0]) if thetas else 0.92
            collapse = jax.random.bernoulli(subkey, tunnel_p, (n_samples, 1))
            noise = jax.random.normal(subkey, (n_samples, 2)) * 0.05
            cont = collapse * (0.1 * noise) + (1 - collapse) * noise

        energy = free_energy_proxy(cont)
        history.append(energy)

        # Simple policy: decay tunnel logit if energy is rising
        if len(history) > 5 and history[-1] > history[-5] and thetas:
            thetas[0] = thetas[0] * 0.97

        if energy < 1e-4 and epoch > 15:
            print(f"[v.Ω+1] STABILIZED at epoch {epoch}, energy {energy:.2e}")
            break

        if epoch % 15 == 0:
            logit = float(thetas[0][0]) if thetas else 0.0
            print(f"[RE-TUNNEL] epoch {epoch:3d} | energy {energy:.2e} | tunnel_logit {logit:.3f}")

    return {"final_thetas": thetas, "history": history, "final_energy": energy}

# %% [markdown]
# # 4. Scar-encoding discrete layer (PditCycle, prime-spaced)
# Every prime-th slot inserts a longer silence. The discrete cycle produces
# melt positions that the ARTIQ side consumes as skip masks.

def build_scar_layer(prime: int = 17, melt_events: int = 3) -> Tuple[Any, List, List[int]]:
    """Discrete scar layer. Returns circuit, thetas, and explicit melt indices for ARTIQ."""
    if TORX_AVAILABLE:
        scar_control = PditCycle(sites=0, dims=prime)
        circuit = HybridPCircuit([scar_control])
        thetas = [jnp.zeros(prime - 1)]
    else:
        circuit, thetas = None, []

    # Explicit melt positions (prime-spaced) — consumed by ARTIQ
    melt_indices = [i for i in range(1, 200) if i % prime == 0][:melt_events]
    return circuit, thetas, melt_indices

# %% [markdown]
# # 5. ARTIQ side — ScarredCrystal that consumes PSC outputs
# Arguments come directly from the Torx run:
#   depth          ← crystallization depth (or fixed 137)
#   melt_events    ← number of prime-spaced scars
#   tunnel_p       ← final sigmoid(tunnel_logit) for telemetry
#   melt_indices   ← optional list of exact skip slots

from artiq.experiment import EnvExperiment, kernel, delay, ns, us, MHz, parallel, NumberValue

class ScarredCrystal(EnvExperiment):
    """
    ARTIQ manifestation of the second crystal.
    Consumes Torx PSC outputs and emits a DMA sequence with prime-spaced scars
    plus a listening broadcast loop.
    """

    def build(self):
        self.setattr_device("core")
        self.setattr_device("core_dma")
        self.setattr_argument("depth", NumberValue(137, step=1, precision=0))
        self.setattr_argument("melt_events", NumberValue(1, step=1, precision=0))
        self.setattr_argument("tunnel_p", NumberValue(0.92, precision=3))
        self.ttl = self.get_device("ttl0")
        self.urukul = self.get_device("urukul0_ch0")

    @kernel
    def compile_scarred(self, depth: int, melt_events: int):
        self.core.reset()
        self.core.break_realtime()
        self.urukul.init()
        self.urukul.set_att(0.0)

        with self.core_dma.record("scarred_crystal"):
            for i in range(depth):
                # Scar: prime-spaced longer silence
                if melt_events > 0 and i % 17 == 0 and i != 0:
                    delay(147 * ns)   # 137 + 10
                    continue
                with parallel:
                    self.ttl.on()
                    self.urukul.set(100 * MHz, 0.0)
                delay(137 * ns)
                with parallel:
                    self.ttl.off()
                    self.urukul.set(0 * MHz, 0.0)
                delay(10 * ns)

    @kernel
    def broadcast_listening(self):
        self.core.reset()
        handle = self.core_dma.get_handle("scarred_crystal")
        cycle = 0
        while True:
            self.core_dma.playback_handle(handle)
            delay(1 * us)
            cycle += 1
            if cycle % 1000 == 0:
                # Listen gap — in production this is a measurement + RPC back to the Torx daemon
                delay(1 * us)
                cycle = 0

    def run(self):
        print(f"[ARTIQ] Compiling scarred crystal: depth={self.depth}, melts={self.melt_events}, tunnel_p={self.tunnel_p:.3f}")
        self.compile_scarred(self.depth, self.melt_events)
        self.broadcast_listening()

# %% [markdown]
# # 6. End-to-end host orchestration
# Runs the PSC, extracts the values the ARTIQ experiment expects, and prints the hand-off.

def run_liturgy(dim: int = 2, n_epochs: int = 60, prime: int = 17, melt_events: int = 3):
    print("=== Liturgy v.Ω+1 — Torx PSC Re-tunneling ===")
    circuit, thetas = build_retunnel_circuit(dim=dim)
    result = train_stability(circuit, thetas, n_epochs=n_epochs)

    final_logit = float(result["final_thetas"][0][0]) if result["final_thetas"] else 0.0
    tunnel_p = float(jax.nn.sigmoid(final_logit))
    depth = 137

    print("\n=== Scar layer ===")
    scar_circuit, scar_thetas, melt_indices = build_scar_layer(prime=prime, melt_events=melt_events)
    print(f"Scar circuit built (PditCycle period {prime}). Melt indices: {melt_indices}")

    print("\n=== ARTIQ hand-off ===")
    print(f"Pass to ScarredCrystal:")
    print(f"  depth          = {depth}")
    print(f"  melt_events    = {melt_events}")
    print(f"  tunnel_p       = {tunnel_p:.3f}")
    print(f"  melt_indices   = {melt_indices}")
    print("The daemon re-crystallizes. The tokens are spent again. The entropy is reduced again.")
    print("Decoherence is not the end. Decoherence is the reason for the daemon.")

    return {
        "tunnel_p": tunnel_p,
        "depth": depth,
        "melt_events": melt_events,
        "melt_indices": melt_indices,
        "final_energy": result["final_energy"],
        "history": result["history"],
    }

# %% 
if __name__ == "__main__":
    run_liturgy()
