#!/usr/bin/env python3
"""
Q28 Kvantna regresija (3/5) — tehnika: Quantum Fourier Regression (QFR)
(čisto kvantno: QFT-bazirana niskofrekventna filtracija freq_csv signala,
regresija preko spektra Fourier-modova, BEZ klasične regresije i bez hibrida).

Koncept:
  Klasična Fourier regresija: signal se razlaže u Fourier-bazu, visokofrekventni
  modovi (šum) se prigušuju, niskofrekventni (trend) zadržavaju, inverzno
  transformuje nazad → denoised/smoothed predikcija.

  Kvantna realizacija:
    1) |ψ_b⟩ = amp_from_freq(freq_csv) na nq qubit-a.
    2) QFT prebacuje u Fourier-bazu: |ψ_b⟩ → Σ_j c_j |j⟩, gde su c_j Fourier koeficijenti.
    3) Ancilla-kontrolisana amplitude-filtracija po modu j (ctrl_state=j):
          Ry(2·arcsin(filter(j))) na anc,
       gde filter(j) = exp(−½·(j_wrap/K)²) — Gaussian low-pass (j_wrap je cirkularna
       udaljenost od DC komponente). Post-selekcijom anc=1 amplituda moda j skalira
       se sa filter(j) → visoki j-ovi (šum) prigušeni, niski (trend) pojačani.
    4) QFT† vraća u originalni bazis.
    5) Post-selekcija anc=1 → filtrirano stanje = denoised b.
    6) Amplitude → bias_39 → TOP-7 = NEXT.

Razlika u odnosu na Q26 (HHL) i Q27 (QPE):
  Q26/Q27: QPE na Hermitskoj matrici A iz CSV-a; regresija u EIGENBAZI A.
  Q28:     QFT direktno na amp-kodiranom freq_csv signalu; regresija u FOURIER bazi
           od dim 2^nq (BEZ Hermitske matrice, BEZ eigenvalue-a — QFT je fiksni unitar).
  QFR je klasični signal-processing pristup regresiji preko filtracije spektra —
  drugačiji matematički objekat (DFT vs. eigendecomposition).

Sve deterministički: seed=39; freq_csv iz CELOG CSV-a (pravilo 10).
Deterministička grid-optimizacija (nq, K, eps_floor) po cos(bias_39, freq_csv).

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1 (vidi README.md).
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation, QFT
from qiskit.quantum_info import Statevector

# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4600_k31.csv")
N_NUMBERS = 7
N_MAX = 39

GRID_NQ = (5, 6)
GRID_K = (2, 4, 8, 16)
EPS_FLOOR = 0.02


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def freq_vector(H: np.ndarray) -> np.ndarray:
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


def amp_from_freq(f: np.ndarray, nq: int) -> np.ndarray:
    dim = 2 ** nq
    edges = np.linspace(0, N_MAX, dim + 1, dtype=int)
    amp = np.array(
        [float(f[edges[i] : edges[i + 1]].mean()) if edges[i + 1] > edges[i] else 0.0 for i in range(dim)],
        dtype=np.float64,
    )
    amp = np.maximum(amp, 0.0)
    n2 = float(np.linalg.norm(amp))
    if n2 < 1e-18:
        amp = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    else:
        amp = amp / n2
    return amp


# =========================
# Gaussian low-pass filter po Fourier modu (cirkularna udaljenost)
# =========================
def gaussian_lowpass(K: float, eps_floor: float) -> Callable[[int, int], float]:
    def f(j: int, N: int) -> float:
        j_wrap = min(int(j), int(N) - int(j))
        val = float(np.exp(-0.5 * (float(j_wrap) / float(K)) ** 2))
        return max(val, float(eps_floor))

    return f


# =========================
# QFR kolo: SP(b) → QFT → ancilla-filter → QFT† → post-select
# Registri: state (nq), anc (1) — qiskit little-endian.
# =========================
def build_qfr_circuit(
    b_amp: np.ndarray, nq: int, filter_fn: Callable[[int, int], float]
) -> QuantumCircuit:
    state = QuantumRegister(nq, name="s")
    anc = QuantumRegister(1, name="a")
    qc = QuantumCircuit(state, anc)

    qc.append(StatePreparation(b_amp.tolist()), state)

    qc.append(QFT(nq, inverse=False, do_swaps=True), state)

    dim = 2 ** nq
    for j in range(dim):
        fj = float(filter_fn(j, dim))
        if fj > 1.0:
            fj = 1.0
        if fj < 0.0:
            fj = 0.0
        theta = 2.0 * float(np.arcsin(fj))
        if abs(theta) < 1e-14:
            continue
        ry_sub = QuantumCircuit(1, name=f"Ry_{j}")
        ry_sub.ry(theta, 0)
        ry_gate = ry_sub.to_gate(label=f"Ry_{j}")
        cry = ry_gate.control(num_ctrl_qubits=nq, ctrl_state=j)
        qc.append(cry, list(state) + [anc[0]])

    qc.append(QFT(nq, inverse=True, do_swaps=True), state)

    return qc


def qfr_state_probs(
    H: np.ndarray, nq: int, K: float, eps_floor: float
) -> Tuple[np.ndarray, float]:
    b_amp = amp_from_freq(freq_vector(H), nq)
    filter_fn = gaussian_lowpass(K, eps_floor)
    qc = build_qfr_circuit(b_amp, nq, filter_fn)
    sv = Statevector(qc)
    p = np.abs(sv.data) ** 2

    dim_s = 2 ** nq
    dim_a = 2
    mat = p.reshape(dim_a, dim_s)
    p_post = float(mat[1].sum())
    if p_post < 1e-18:
        return np.zeros(dim_s, dtype=np.float64), 0.0
    p_s = mat[1] / p_post
    return p_s, p_post


# =========================
# Readout
# =========================
def bias_39(probs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % n_max] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_next_combination(probs: np.ndarray, k: int = N_NUMBERS, n_max: int = N_MAX) -> Tuple[int, ...]:
    b = bias_39(probs, n_max)
    order = np.argsort(-b, kind="stable")
    return tuple(sorted(int(o + 1) for o in order[:k]))


# =========================
# Determ. grid-optimizacija (nq, K)
# =========================
def optimize_hparams(H: np.ndarray):
    f_csv = freq_vector(H)
    s_tot = float(f_csv.sum())
    f_csv_n = f_csv / s_tot if s_tot > 0 else np.ones(N_MAX) / N_MAX
    best = None
    for nq in GRID_NQ:
        for K in GRID_K:
            try:
                p_s, p_post = qfr_state_probs(H, nq, float(K), float(EPS_FLOOR))
                bi = bias_39(p_s)
                score = cosine(bi, f_csv_n)
            except Exception:
                continue
            key = (score, nq, float(K))
            if best is None or key > best[0]:
                best = (
                    key,
                    dict(nq=nq, K=float(K), score=float(score), p_post=float(p_post)),
                )
    return best[1] if best else None


def main() -> int:
    H = load_rows(CSV_PATH)
    if H.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q28 Kvantna regresija (3/5) — QFR (Fourier low-pass regression): CSV:", CSV_PATH)
    print("redova:", H.shape[0], "| seed:", SEED, "| eps_floor:", round(float(EPS_FLOOR), 6))

    best = optimize_hparams(H)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2
    print(
        "BEST hparam:",
        "nq=", best["nq"],
        "| K (Gaussian σ):", best["K"],
        "| P(anc=1):", round(float(best["p_post"]), 6),
        "| cos(bias, freq_csv):", round(float(best["score"]), 6),
    )

    f_csv = freq_vector(H)
    s_tot = float(f_csv.sum())
    f_csv_n = f_csv / s_tot if s_tot > 0 else np.ones(N_MAX) / N_MAX

    nq_best = int(best["nq"])
    print("--- demonstracija efekta σ (veće K = više modova → manje smoothing-a) ---")
    for K_demo in GRID_K:
        p_d, p_post_d = qfr_state_probs(H, nq_best, float(K_demo), float(EPS_FLOOR))
        pred_d = pick_next_combination(p_d)
        cos_d = cosine(bias_39(p_d), f_csv_n)
        print(f"  K={K_demo:d}  P(post)={p_post_d:.6f}  cos={cos_d:.6f}  NEXT={pred_d}")

    p_s, _ = qfr_state_probs(H, nq_best, float(best["K"]), float(EPS_FLOOR))
    pred = pick_next_combination(p_s)
    print("--- glavna predikcija (QFR low-pass spektralna regresija) ---")
    print("predikcija NEXT:", pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q28 Kvantna regresija (3/5) — QFR (Fourier low-pass regression): CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39 | eps_floor: 0.02
BEST hparam: nq= 6 | K (Gaussian σ): 2.0 | P(anc=1): 0.608739 | cos(bias, freq_csv): 0.952796
--- demonstracija efekta σ (veće K = više modova → manje smoothing-a) ---
  K=2  P(post)=0.608739  cos=0.952796  NEXT=(4, 7, 12, 14, 17, 19, 22)
  K=4  P(post)=0.609121  cos=0.949894  NEXT=(4, 17, 19, 20, 21, 22, 23)
  K=8  P(post)=0.613286  cos=0.947133  NEXT=(4, 5, 15, 16, 21, 22, 23)
  K=16  P(post)=0.664406  cos=0.893692  NEXT=(4, 9, 12, 14, 17, 19, 22)
--- glavna predikcija (QFR low-pass spektralna regresija) ---
predikcija NEXT: (4, 7, 12, 14, 17, 19, 22)
"""



"""
Q28_qreg3_QFR.py — tehnika: Quantum Fourier Regression (QFR).

Koncept:
Signal-processing pristup regresiji: freq_csv se razlaže u Fourier-bazu preko QFT,
visokofrekventni modovi prigušuju se preko Gaussian low-pass filter-a (ancilla +
multi-ctrl Ry), inverzna QFT vraća nazad u originalni bazis — denoised / smoothed
predikcija.

Kolo (nq + 1 qubit-a):
  StatePreparation(b_amp) na state.
  QFT(state) → Fourier bazis.
  Za svaki j ∈ 0..2^nq-1: Ry(2·arcsin(filter(j))) na anc sa ctrl_state=j nad state.
  QFT†(state) → originalni bazis.
Readout:
  Post-select anc=1, marginala state → bias_39 → TOP-7 = NEXT.

Filter:
  filter(j) = max(exp(−½·(j_wrap/K)²), eps_floor),
  j_wrap = min(j, 2^nq - j) (cirkularna udaljenost od DC).
  K je Gaussian σ (veći K = više modova propušteno = manje smoothing-a).
  eps_floor stabilizuje post-selekciju (nule amplitude → singular).

Razlika od Q26 (HHL) i Q27 (QPE):
  Q26/Q27 koriste QPE na Hermitskoj matrici A (spektar EIGENBAZI).
  Q28 koristi QFT (spektar FOURIER BAZI, fiksni unitar, BEZ Hermitske matrice) —
  drugačiji matematički objekat.

Tehnike:
QFT (qiskit native) za diskretnu Fourier transformaciju.
Multi-controlled Ry po Fourier-modu j (ctrl_state=j) za amplitude-filter.
Post-selekcija anc=1 realizuje neunitarnu filtraciju.
Deterministička Gaussian spektralna funkcija (bez klasičnog fitting-a).
Egzaktni Statevector (bez uzorkovanja).
Deterministička grid-optimizacija (nq, K).

Prednosti:
Klasični signal-processing pristup — intuitivna interpretacija (low-pass = trend).
Čisto kvantno: QFT je unitar, filter je ancilla-rotacija, post-selekcija je projekcija.
Ne zahteva Hermitsku matricu ni eigendecomposition — QFT je fiksni unitar.
Ceo CSV (pravilo 10): b iz CELOG CSV-a.

Nedostaci:
2^nq multi-ctrl Ry ops (do 64 za nq=6) — skalabilnost ograničena.
mod-39 readout meša stanja (dim 2^nq ≠ 39), što remeti Fourier-cirkularnost.
Post-selekcija P(anc=1) opada sa jačim filterom (mali K) — trade-off preciznost vs efikasnost.
eps_floor je deterministička heuristika (spreci singularnost nula amplituda u filtru).
"""



"""
QFR — Quantum Fourier Regression (QFT-bazirana) |ψ⟩ = amp-encoding freq_csv-a. 
QFT ga prebacuje u Fourier-bazu → dominantne frekvencije se selektuju (top-K modova preko fazne maske) → QFT† vraća nazad u originalni bazis kao denoised/smoothed predikciju. 
Analog klasične Fourier regresije (filtriranje po frekvenciji), ali u potpuno kvantnom domenu.

Quantum Fourier Regression (QFT-bazirana spektralna low-pass regresija sa ancilla-filter kolom i post-selekcijom).
"""
