"""
treino.py — Desafio 3: o melhor treino para o nosso problema (HAR70+)
GBC073 — Inteligência Computacional (FACOM/UFU)

Treina uma CNN 1D pequena em janelas de 2 s dos acelerômetros (costas + coxa) e registra,
a cada k passos, os sinais do treino: perda, norma do gradiente (antes de qualquer clipping),
norma dos parâmetros, razão de atualização ‖θ_{t+1}−θ_t‖/‖θ_t‖ e a taxa efetiva η_t — tanto
globalmente quanto por camada (conv1 = primeira, conv2/conv3/fc = meio, head = última).

Tudo é configurável pela linha de comando (otimizador, η, agenda, semente). Exemplo:

    python treino.py --otimizador adamw --lr 3e-3 --agenda warmup_cosseno --semente 0

Cada execução grava logs/<id>.csv (um registro a cada k passos) e imprime o resumo em JSON.
Qualquer linha de experimentos.csv é reproduzível com o comando da coluna `comando`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AQUI = os.path.dirname(os.path.abspath(__file__))
DIR_LOGS = os.path.join(AQUI, "logs")

OTIMIZADORES = ["sgd", "nesterov", "adam", "adamw", "muon", "nadam"]
AGENDAS = ["constante", "cosseno", "warmup_cosseno", "wsd"]
CAMADAS = ["conv1.weight", "conv2.weight", "conv3.weight", "fc.weight", "head.weight"]
N_CLASSES = 7

# hiperparâmetros fixos (iguais em todas as comparações)
PADRAO = dict(passos=2000, lote=128, k=25, aval_cada=200, momento=0.9,
              wd_sgd=5e-4, wd_adamw=0.01, wd_muon=0.01, momento_muon=0.95,
              lr_adamw_muon=3e-3, warmup=0.05, decaimento_wsd=0.2, clip=0.0)


# =============================================================================
# Modelo
# =============================================================================
class Rede(nn.Module):
    """CNN 1D: 3 convoluções (BN + ReLU) -> média temporal -> fc oculta -> head (logits)."""

    def __init__(self, c_in=6, n_classes=N_CLASSES, largura=32):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, 16, 7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, largura, 5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(largura)
        self.conv3 = nn.Conv1d(largura, largura, 5, stride=2, padding=2, bias=False)
        self.bn3 = nn.BatchNorm1d(largura)
        self.fc = nn.Linear(largura, largura)
        self.head = nn.Linear(largura, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=2)
        x = F.relu(self.fc(x))
        return self.head(x)


# =============================================================================
# Muon (implementação própria: aceita kernels de convolução achatados)
# =============================================================================
def ortogonaliza(G: torch.Tensor, passos: int = 5) -> torch.Tensor:
    """Newton–Schulz quíntico (Jordan, 2024): devolve ≈ U V^T de G = U Σ V^T."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    T = X.size(0) > X.size(1)
    if T:
        X = X.T
    for _ in range(passos):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if T else X


class Muon(torch.optim.Optimizer):
    """Momento (Nesterov) + ortogonalização do passo. Para W de forma (C_out, ...), o passo é
    feito sobre a matriz achatada (C_out, C_in·k). Escala max(1, linhas/colunas)^0,5 como no
    Muon original; weight decay desacoplado."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0, ns_passos=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay, ns_passos=ns_passos))

    @torch.no_grad()
    def step(self):
        for grupo in self.param_groups:
            lr, mom, wd = grupo["lr"], grupo["momentum"], grupo["weight_decay"]
            for p in grupo["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"]
                buf.lerp_(p.grad, 1 - mom)
                g = p.grad.lerp(buf, mom) if grupo["nesterov"] else buf
                G = g.reshape(g.size(0), -1)
                O = ortogonaliza(G, grupo["ns_passos"])
                escala = max(1.0, G.size(0) / G.size(1)) ** 0.5
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(O.view_as(p), alpha=-lr * escala)


class Combinado:
    """Vários otimizadores agindo como um só (Muon nas matrizes ocultas + AdamW no resto)."""

    def __init__(self, opts):
        self.opts = opts

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()


def separa_muon(modelo):
    """Muon: pesos de conv/fc ocultas (ndim ≥ 2, fora a head). AdamW: vieses, BN, head."""
    muon_p, nomes = [], []
    for n, p in modelo.named_parameters():
        if p.ndim >= 2 and not n.startswith("head"):
            muon_p.append(p); nomes.append(n)
    ids = {id(p) for p in muon_p}
    adamw_p = [p for p in modelo.parameters() if id(p) not in ids]
    return muon_p, adamw_p, nomes


def cria_otimizador(nome, modelo, lr, cfg):
    """Devolve (otimizador com .step/.zero_grad, lista de otimizadores torch para as agendas)."""
    P = modelo.parameters()
    if nome == "sgd":
        o = torch.optim.SGD(P, lr=lr, momentum=cfg["momento"], weight_decay=cfg["wd_sgd"])
    elif nome == "nesterov":
        o = torch.optim.SGD(P, lr=lr, momentum=cfg["momento"], nesterov=True, weight_decay=cfg["wd_sgd"])
    elif nome == "adam":
        o = torch.optim.Adam(P, lr=lr)
    elif nome == "adamw":
        o = torch.optim.AdamW(P, lr=lr, weight_decay=cfg["wd_adamw"])
    elif nome == "nadam":
        o = torch.optim.NAdam(P, lr=lr)
    elif nome == "muon":
        muon_p, adamw_p, _ = separa_muon(modelo)
        opts = [Muon(muon_p, lr=lr, momentum=cfg["momento_muon"], weight_decay=cfg["wd_muon"]),
                torch.optim.AdamW(adamw_p, lr=cfg["lr_adamw_muon"], weight_decay=cfg["wd_adamw"])]
        return Combinado(opts), opts
    else:
        raise ValueError(f"otimizador desconhecido: {nome}")
    return o, [o]


# =============================================================================
# Agendas de η (fator multiplicativo; a MESMA agenda em todos os otimizadores do Combinado)
# =============================================================================
def fator_agenda(nome, T, warmup=0.05, decaimento=0.2):
    W = max(1, int(round(warmup * T)))
    D = max(1, int(round(decaimento * T)))

    def constante(t):
        return 1.0

    def cosseno(t):
        return 0.5 * (1 + math.cos(math.pi * min(t, T) / T))

    def warmup_cosseno(t):
        if t < W:
            return (t + 1) / W
        return 0.5 * (1 + math.cos(math.pi * min(t - W, T - W) / (T - W)))

    def wsd(t):  # warmup–stable–decay (Hägele et al., 2024): decaimento linear nos últimos 20%
        if t < W:
            return (t + 1) / W
        if t < T - D:
            return 1.0
        return max(0.0, (T - t) / D)

    return dict(constante=constante, cosseno=cosseno, warmup_cosseno=warmup_cosseno, wsd=wsd)[nome]


# =============================================================================
# Dados (cache por processo)
# =============================================================================
_DADOS = {}


def dados():
    if not _DADOS:
        from dados import carregar
        d = carregar()
        for parte in ("treino", "val", "teste"):
            _DADOS[f"X_{parte}"] = torch.from_numpy(d[f"X_{parte}"])
            _DADOS[f"y_{parte}"] = torch.from_numpy(d[f"y_{parte}"])
        cont = torch.bincount(_DADOS["y_treino"], minlength=N_CLASSES).float()
        w = (cont.sum() / cont).sqrt()                 # pesos ∝ 1/√freq (classes raras: escada)
        _DADOS["peso"] = w / w.mean()
    return _DADOS


@torch.no_grad()
def avaliar(modelo, X, y, peso):
    modelo.eval()
    logits = torch.cat([modelo(X[i:i + 2048]) for i in range(0, len(X), 2048)])
    modelo.train()
    perda = F.cross_entropy(logits, y, weight=peso).item()
    pred = logits.argmax(1)
    acc = (pred == y).float().mean().item()
    f1s = []
    for c in range(N_CLASSES):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return dict(perda=perda, acc=acc, f1=float(np.mean(f1s)), f1_classes=f1s)


# =============================================================================
# Uma execução
# =============================================================================
def id_execucao(c):
    base = f"{c['etapa']}_{c['otimizador']}_lr{c['lr']:.1e}_{c['agenda']}_s{c['semente']}"
    if c["otimizador"] == "muon":
        base += f"_aw{c['lr_adamw_muon']:.1e}"
    return base


def comando(c):
    s = (f"python treino.py --otimizador {c['otimizador']} --lr {c['lr']:g} --agenda {c['agenda']} "
         f"--semente {c['semente']} --passos {c['passos']} --lote {c['lote']}")
    if c["otimizador"] == "muon":
        s += f" --lr-adamw-muon {c['lr_adamw_muon']:g}"
    if c.get("teste"):
        s += " --teste"
    return s + f" --etapa {c['etapa']}"


def treinar(cfg: dict) -> dict:
    c = dict(PADRAO); c.update(cfg)
    c.setdefault("etapa", "manual"); c.setdefault("teste", False)
    rid = c.get("id") or id_execucao(c)
    D = dados()
    Xtr, ytr, peso = D["X_treino"], D["y_treino"], D["peso"]

    torch.manual_seed(c["semente"]); np.random.seed(c["semente"])
    modelo = Rede()
    opt, opts_torch = cria_otimizador(c["otimizador"], modelo, c["lr"], c)
    fat = fator_agenda(c["agenda"], c["passos"], c["warmup"], c["decaimento_wsd"])
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, fat) for o in opts_torch]

    params = dict(modelo.named_parameters())
    g_amostra = torch.Generator().manual_seed(c["semente"])
    ordem, pos = torch.randperm(len(Xtr), generator=g_amostra), 0

    log, perdas_janela, gmax_janela = [], [], 0.0
    divergiu, t0 = False, time.perf_counter()
    perda0 = None
    for t in range(c["passos"]):
        if pos + c["lote"] > len(ordem):
            ordem, pos = torch.randperm(len(Xtr), generator=g_amostra), 0
        idx = ordem[pos:pos + c["lote"]]; pos += c["lote"]

        perda = F.cross_entropy(modelo(Xtr[idx]), ytr[idx], weight=peso)
        opt.zero_grad()
        perda.backward()
        pv = perda.item()
        perda0 = pv if perda0 is None else perda0
        # norma total do gradiente ANTES do clipping (clip_grad_norm_ devolve a norma pré-corte)
        g = torch.nn.utils.clip_grad_norm_(modelo.parameters(),
                                           c["clip"] if c["clip"] > 0 else float("inf")).item()
        perdas_janela.append(pv); gmax_janela = max(gmax_janela, g)

        if not math.isfinite(pv) or not math.isfinite(g) or pv > 10 * max(perda0, 1.0):
            divergiu = True
            log.append(dict(passo=t, perda=pv, perda_lote=pv, grad=g, grad_max=gmax_janela,
                            divergiu=1))
            break

        registrar = (t % c["k"] == 0) or (t == c["passos"] - 1)
        if registrar:
            linha = dict(passo=t, perda=float(np.mean(perdas_janela)), perda_lote=pv, grad=g,
                         grad_max=gmax_janela)
            with torch.no_grad():
                linha["theta"] = torch.sqrt(sum((p ** 2).sum() for p in modelo.parameters())).item()
                for n in CAMADAS:
                    p = params[n]
                    linha[f"grad/{n}"] = p.grad.norm().item()
                    linha[f"theta/{n}"] = p.norm().item()
            antes = {n: params[n].detach().clone() for n in CAMADAS}
            todos_antes = [p.detach().clone() for p in modelo.parameters()]
            linha["lr"] = opts_torch[0].param_groups[0]["lr"]
            if len(opts_torch) > 1:
                linha["lr_adamw"] = opts_torch[1].param_groups[0]["lr"]

        opt.step()
        for s in scheds:
            s.step()

        if registrar:
            with torch.no_grad():
                for n in CAMADAS:
                    linha[f"rho/{n}"] = ((params[n] - antes[n]).norm() / antes[n].norm()).item()
                num = math.sqrt(sum(((p - a) ** 2).sum().item() for p, a in zip(modelo.parameters(), todos_antes)))
                linha["rho"] = num / linha["theta"]
            if t % c["aval_cada"] == 0 or t == c["passos"] - 1:
                av = avaliar(modelo, D["X_val"], D["y_val"], peso)
                linha.update(val_perda=av["perda"], val_f1=av["f1"], val_acc=av["acc"])
            log.append(linha)
            perdas_janela, gmax_janela = [], 0.0

    tempo = time.perf_counter() - t0
    res = dict(id=rid, etapa=c["etapa"], otimizador=c["otimizador"], lr=c["lr"],
               lr_adamw_muon=c["lr_adamw_muon"] if c["otimizador"] == "muon" else "",
               agenda=c["agenda"], semente=c["semente"], passos=c["passos"], lote=c["lote"],
               passos_feitos=t + 1, divergiu=divergiu, tempo_s=round(tempo, 1))
    if divergiu:
        res.update(val_f1=float("nan"), val_acc=float("nan"), val_perda=float("nan"))
    else:
        av = avaliar(modelo, D["X_val"], D["y_val"], peso)
        res.update(val_f1=round(av["f1"], 5), val_acc=round(av["acc"], 5), val_perda=round(av["perda"], 5))
        if c["teste"]:
            te = avaliar(modelo, D["X_teste"], D["y_teste"], peso)
            res.update(teste_f1=round(te["f1"], 5), teste_acc=round(te["acc"], 5),
                       teste_f1_classes=[round(v, 4) for v in te["f1_classes"]])
    res["comando"] = comando(c)
    res["log"] = f"logs/{rid}.csv"

    os.makedirs(DIR_LOGS, exist_ok=True)
    import csv
    chaves = []
    for l in log:
        for k_ in l:
            if k_ not in chaves:
                chaves.append(k_)
    with open(os.path.join(DIR_LOGS, f"{rid}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=chaves)
        w.writeheader()
        for l in log:
            w.writerow(l)
    return res


def main():
    ap = argparse.ArgumentParser(description="Treino instrumentado — Desafio 3 (HAR70+)")
    ap.add_argument("--otimizador", choices=OTIMIZADORES, default="adamw")
    ap.add_argument("--lr", type=float, default=3e-3, help="η (no Muon: η das matrizes ocultas)")
    ap.add_argument("--lr-adamw-muon", type=float, default=PADRAO["lr_adamw_muon"],
                    help="η da parte AdamW quando --otimizador muon")
    ap.add_argument("--agenda", choices=AGENDAS, default="warmup_cosseno")
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--passos", type=int, default=PADRAO["passos"])
    ap.add_argument("--lote", type=int, default=PADRAO["lote"])
    ap.add_argument("--clip", type=float, default=0.0, help="0 = sem clipping (a norma é registrada igual)")
    ap.add_argument("--teste", action="store_true", help="avalia no teste ao final (usar UMA vez)")
    ap.add_argument("--etapa", default="manual", help="rótulo da etapa do protocolo (entra no id)")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    res = treinar(dict(otimizador=a.otimizador, lr=a.lr, lr_adamw_muon=a.lr_adamw_muon,
                       agenda=a.agenda, semente=a.semente, passos=a.passos, lote=a.lote,
                       clip=a.clip, teste=a.teste, etapa=a.etapa))
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
