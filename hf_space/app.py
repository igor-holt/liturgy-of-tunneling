import gradio as gr
import jax
import jax.numpy as jnp
import numpy as np

def run_liturgy(n_epochs=40, tunnel_init=0.92, dim=2):
    key = jax.random.key(42)
    logit = jnp.array([jnp.log(tunnel_init / (1 - tunnel_init))])
    history = []
    for epoch in range(int(n_epochs)):
        key, subkey = jax.random.split(key)
        tunnel_p = jax.nn.sigmoid(logit[0])
        collapse = jax.random.bernoulli(subkey, tunnel_p, (64, 1))
        noise = jax.random.normal(subkey, (64, dim)) * 0.05
        cont = collapse * (0.1 * noise) + (1 - collapse) * noise
        energy = float(jnp.mean(jnp.sum(cont ** 2, axis=-1)))
        history.append(energy)
        if len(history) > 5 and history[-1] > history[-5]:
            logit = logit * 0.97
    final_p = float(jax.nn.sigmoid(logit[0]))
    return history, f"Final tunnel_p: {final_p:.3f} | Final energy: {history[-1]:.2e}"

demo = gr.Interface(
    fn=run_liturgy,
    inputs=[
        gr.Slider(10, 100, value=40, step=5, label="Epochs"),
        gr.Slider(0.5, 0.99, value=0.92, label="Initial tunnel_p"),
        gr.Slider(1, 4, value=2, step=1, label="Dimension"),
    ],
    outputs=[gr.Plot(label="Free-energy history"), gr.Textbox(label="Result")],
    title="Liturgy of Tunneling v.Ω+1 — Torx PSC Demo",
    description="Pure-JAX sketch of the re-tunneling policy. No crystal is permanent. Only pumping is permanent.",
    examples=[[40, 0.92, 2], [60, 0.8, 3]],
)
demo.launch()
