"""
dados.py — Janelamento da HAR70+ e divisão por participante (Desafio 3, GBC073)

Janelas de 2 s (100 amostras a 50 Hz). Rótulo = classe majoritária da janela; janelas com
pureza < 80% são descartadas (transições). Treino com passo de 0,5 s (75% de sobreposição),
validação e teste com passo de 1 s. A divisão é POR PARTICIPANTE (ninguém aparece em dois
conjuntos), escolhida para que os três conjuntos tenham exemplos de escada (classes 4 e 5),
que são raras: 10 participantes no treino, 4 na validação, 4 no teste.

O resultado fica em cache em dados_janelas.npz (gerado na primeira execução).
"""
from __future__ import annotations

import os

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
DIR_CSV = os.path.join(AQUI, "har70", "har70plus")
CACHE = os.path.join(AQUI, "dados_janelas.npz")

CANAIS = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
ROTULOS = [1, 3, 4, 5, 6, 7, 8]
NOMES = ["caminhar", "arrastar pés", "subir escada", "descer escada", "em pé", "sentado", "deitado"]
JANELA = 100           # 2 s a 50 Hz
PASSO_TREINO = 25      # 0,5 s
PASSO_AVAL = 50        # 1 s
PUREZA_MIN = 0.8

PARTICIPANTES = {
    "treino": [501, 502, 503, 505, 506, 509, 510, 512, 513, 518],
    "val": [504, 508, 511, 514],
    "teste": [507, 515, 516, 517],
}


def _janelar(arq: str, passo: int):
    import pandas as pd
    d = pd.read_csv(arq, usecols=CANAIS + ["label"])
    x = d[CANAIS].to_numpy(np.float32)
    y = d["label"].to_numpy()
    mapa = {r: i for i, r in enumerate(ROTULOS)}
    yi = np.vectorize(mapa.get)(y)
    Xs, ys = [], []
    for ini in range(0, len(x) - JANELA + 1, passo):
        seg = yi[ini:ini + JANELA]
        cont = np.bincount(seg, minlength=len(ROTULOS))
        c = int(cont.argmax())
        if cont[c] / JANELA < PUREZA_MIN:
            continue
        Xs.append(x[ini:ini + JANELA].T)
        ys.append(c)
    return np.stack(Xs), np.array(ys, dtype=np.int64)


def preparar(forcar: bool = False) -> None:
    if os.path.exists(CACHE) and not forcar:
        return
    saida = {}
    for parte, ids in PARTICIPANTES.items():
        passo = PASSO_TREINO if parte == "treino" else PASSO_AVAL
        Xs, ys, ps = [], [], []
        for pid in ids:
            X, y = _janelar(os.path.join(DIR_CSV, f"{pid}.csv"), passo)
            Xs.append(X); ys.append(y); ps.append(np.full(len(y), pid))
        saida[f"X_{parte}"] = np.concatenate(Xs)
        saida[f"y_{parte}"] = np.concatenate(ys)
        saida[f"p_{parte}"] = np.concatenate(ps)
    # padronização por canal com estatísticas SÓ do treino
    mu = saida["X_treino"].mean(axis=(0, 2), keepdims=True)
    sd = saida["X_treino"].std(axis=(0, 2), keepdims=True) + 1e-6
    for parte in PARTICIPANTES:
        saida[f"X_{parte}"] = ((saida[f"X_{parte}"] - mu) / sd).astype(np.float32)
    np.savez_compressed(CACHE, **saida)


def carregar():
    preparar()
    z = np.load(CACHE)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    preparar(forcar=True)
    d = carregar()
    for parte in PARTICIPANTES:
        y = d[f"y_{parte}"]
        cont = np.bincount(y, minlength=len(ROTULOS))
        print(f"{parte:>6}: {len(y):6d} janelas  " + "  ".join(f"{n}={c}" for n, c in zip(NOMES, cont)))
