"""
Desafio 2 - IC
Nomes: Felipe Canabarro Graffunder
    Isabela de Paula Baborsa
"""

import math
import torch


# Constantes da transformacao; nao sao parametros aprendidos.
_ESCALA_ATIVACAO = 0.5
_DESVIO_INICIAL = 0.25
_GANHO = 2.1162457615312227


def ativacao(x: torch.Tensor) -> torch.Tensor:
    return _ESCALA_ATIVACAO * torch.tanh(x)


@torch.no_grad()
def inicializar(
    W: torch.Tensor,
    b: torch.Tensor,
    fan_in: int,
    fan_out: int,
    camada: int,
    n_camadas: int,
) -> None:
    if camada == 1:
        # Entrada padronizada: Var[W*x] ~= 1/16.
        W.normal_(mean=0.0, std=_DESVIO_INICIAL / math.sqrt(fan_in))

    elif camada == n_camadas:
        # A entrada dos logits ja passou pela ativacao anterior.
        # Mantem Var[logits] ~= 1/16; nao aplica tanh na saida do modelo.
        W.normal_(mean=0.0, std=_GANHO / math.sqrt(fan_in))

    else:
        # Em matrizes quadradas, W/gain e ortogonal.
        # Para fan_out > fan_in, a correcao preserva a energia media por saida.
        ganho_retangular = _GANHO * math.sqrt(max(1.0, fan_out / fan_in))
        torch.nn.init.orthogonal_(W, gain=ganho_retangular)

    b.zero_()
