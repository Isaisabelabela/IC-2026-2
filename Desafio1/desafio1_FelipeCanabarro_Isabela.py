"""
desafio1_aluno.py — Desafio 1: Mapa de características para o Perceptron
GBC073 — Inteligência Computacional (FACOM/UFU)

Nomes: Isabela de Paula Barbosa
        Felipe Canabarro Graffunder
"""
import math
import time
import torch

DIM_MAX = 64
SEMENTES = (0, 1, 2)

# =============================================================================
# >>> SUA SUBMISSÃO — edite apenas esta classe <<<
# =============================================================================
class Submissao:
    DIM_MAX = DIM_MAX

    def fit(self, X: torch.Tensor) -> None:
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-8

        Xs = (X - self.mu) / self.sd
        self.d = X.shape[1]

        if self.d == 2:
            x, y = Xs[:, 0], Xs[:, 1]

            P = torch.stack([
                x * x,
                x * y,
                y * y,
                x ** 3,
                x * x * y,
                x * y * y,
                y ** 3,
            ], dim=1)

            self.p_mu = P.mean(0)
            self.p_sd = P.std(0) + 1e-8

            k = min(55, len(Xs))

            norm2 = (Xs * Xs).sum(1)

            first = int(norm2.argmax())

            ids = [first]

            min_d2 = ((Xs - Xs[first]) ** 2).sum(1)

            for _ in range(1, k):
                j = int(min_d2.argmax())

                ids.append(j)

                d2j = ((Xs - Xs[j]) ** 2).sum(1)

                min_d2 = torch.minimum(min_d2, d2j)

            self.centers = Xs[ids].clone()

            A = Xs[:min(300, len(Xs))]

            d2 = torch.cdist(A, A) ** 2

            positivos = d2[d2 > 1e-12]

            if positivos.numel() > 0:
                med2 = positivos.median()
            else:
                med2 = X.new_tensor(1.0)

            self.gamma = 8.0 / med2

        else:
            r2 = (Xs * Xs).sum(1, keepdim=True)

            self.r2_mu = r2.mean(0)
            self.r2_sd = r2.std(0) + 1e-8

            k = max(0, 64 - self.d - 1)

            if k > 0:
                A = Xs[:min(300, len(Xs))]

                d2 = torch.cdist(A, A) ** 2

                positivos = d2[d2 > 1e-12]

                if positivos.numel() > 0:
                    med2 = positivos.median()
                else:
                    med2 = X.new_tensor(2.0)

                sigma = torch.sqrt(med2 / 2).clamp_min(1e-6)

                g = torch.Generator().manual_seed(12345)

                self.W = (
                    torch.randn(
                        self.d,
                        k,
                        generator=g,
                        dtype=X.dtype
                    )
                    / sigma.cpu()
                )

                self.b = (
                    torch.rand(
                        k,
                        generator=g,
                        dtype=X.dtype
                    )
                    * 2
                    * math.pi
                )

                self.W = self.W.to(X.device)
                self.b = self.b.to(X.device)

            else:
                self.W = None
                self.b = None

    def _projetar(self, Xs: torch.Tensor) -> torch.Tensor:

        P = Xs[:, 0:1] * self.W[0:1, :]

        for j in range(1, self.d):
            P = P + Xs[:, j:j + 1] * self.W[j:j + 1, :]

        return P

    def phi(self, X: torch.Tensor) -> torch.Tensor:
        Xs = (X - self.mu) / self.sd

        if self.d == 2:
            x, y = Xs[:, 0], Xs[:, 1]

            P = torch.stack([
                x * x,
                x * y,
                y * y,
                x ** 3,
                x * x * y,
                x * y * y,
                y ** 3,
            ], dim=1)

            P = (P - self.p_mu) / self.p_sd

            diff = (
                Xs[:, None, :]
                - self.centers[None, :, :]
            )

            d2 = (diff * diff).sum(2)

            R = torch.exp(-self.gamma * d2)

            return torch.cat([
                Xs,
                P,
                R
            ], dim=1)

        r2 = (Xs * Xs).sum(1, keepdim=True)

        r2 = (
            (r2 - self.r2_mu)
            / self.r2_sd
        )

        if self.W is None:
            return torch.cat([
                Xs,
                r2
            ], dim=1)

        proj = self._projetar(Xs)

        R = (
            math.sqrt(2 / self.W.shape[1])
            * torch.cos(proj + self.b)
        )

        return torch.cat([
            Xs,
            r2,
            R
        ], dim=1)
# =============================================================================
# Harness (não edite daqui para baixo)
# =============================================================================
def _luas(n, g):
    t = torch.rand(n // 2, generator=g) * math.pi
    X = torch.cat([torch.stack([torch.cos(t), torch.sin(t)], 1),
                   torch.stack([1 - torch.cos(t), 0.5 - torch.sin(t)], 1)])
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    return X + 0.15 * torch.randn(n, 2, generator=g), y

def _circulos(n, g):
    t = torch.rand(n, generator=g) * 2 * math.pi
    r = torch.where(torch.arange(n) < n // 2, 1.0, 0.45)
    X = torch.stack([r * torch.cos(t), r * torch.sin(t)], 1)
    return X + 0.08 * torch.randn(n, 2, generator=g), (torch.arange(n) >= n // 2).float()

def _xor(n, g):
    X = torch.rand(n, 2, generator=g) * 2 - 1
    y = (X[:, 0] * X[:, 1] < 0).float()
    return X + 0.15 * torch.randn(n, 2, generator=g), y

def _espiral(n, g):
    t = torch.sqrt(torch.rand(n // 2, generator=g)) * 3 * math.pi
    a = torch.stack([t * torch.cos(t), t * torch.sin(t)], 1) / 10
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    return torch.cat([a, -a]) + 0.05 * torch.randn(n, 2, generator=g), y

def _esfera(n, g, d=10):
    X = torch.randn(n, d, generator=g)
    r2 = (X ** 2).sum(1)
    return X, (r2 > r2.median()).float()

TAREFAS = {"xor": lambda g: _xor(600, g), "duas_luas": lambda g: _luas(600, g),
           "circulos": lambda g: _circulos(600, g), "espiral": lambda g: _espiral(800, g),
           "esfera_10d": lambda g: _esfera(800, g)}


@torch.no_grad()
def perceptron_pocket(Z, y, epocas=50, eta=1.0, semente=0):
    """Regra de Rosenblatt (w <- w + eta*y*z nos erros) + pocket: guarda o melhor w."""
    Zb = torch.cat([Z, torch.ones(len(Z), 1)], 1)     # viés embutido
    yb = 2 * y - 1
    w = torch.zeros(Zb.shape[1]); melhor_w, melhor_acc = w.clone(), -1.0
    g = torch.Generator().manual_seed(semente)
    for _ in range(epocas):
        for i in torch.randperm(len(Zb), generator=g).tolist():
            if yb[i] * (Zb[i] @ w) <= 0:
                w += eta * yb[i] * Zb[i]
        acc = ((Zb @ w) * yb > 0).float().mean().item()
        if acc > melhor_acc:
            melhor_acc, melhor_w = acc, w.clone()
    return melhor_w


def acuracia_balanceada(y, yhat):
    return torch.stack([(yhat[y == c] == c).float().mean() for c in y.unique()]).mean().item()


@torch.no_grad()
def rodar(sub, gerador, semente, checar=True):
    g = torch.Generator().manual_seed(semente)
    X, y = gerador(g)
    idx = torch.randperm(len(X), generator=g); ntr = int(0.6 * len(X))
    Xtr, ytr, Xte, yte = X[idx[:ntr]], y[idx[:ntr]], X[idx[ntr:]], y[idx[ntr:]]

    torch.manual_seed(semente)
    sub.fit(Xtr)                                       # nunca recebe ytr
    t0 = time.perf_counter(); Ztr = sub.phi(Xtr); dt = time.perf_counter() - t0
    Zte = sub.phi(Xte)
    if checar:                                         # regras do desafio
        d, dl = Xtr.shape[1], Ztr.shape[1]
        assert Ztr.ndim == 2 and Zte.shape[1] == dl, "phi deve devolver (n, d')"
        assert d < dl <= DIM_MAX, f"exige d < d' <= {DIM_MAX}; recebi d={d}, d'={dl}"
        assert torch.isfinite(Ztr).all() and torch.isfinite(Zte).all(), "NaN/Inf na saída de phi"
        assert torch.allclose(sub.phi(Xtr[:20]), Ztr[:20]), "phi não é determinística"
        assert dt * (10_000 / len(Xtr)) < 2.0, "phi lenta demais (limite: 10^4 pontos em 2 s)"

    w = perceptron_pocket(Ztr, ytr, semente=semente)
    yhat = (torch.cat([Zte, torch.ones(len(Zte), 1)], 1) @ w > 0).float()
    return acuracia_balanceada(yte, yhat)


class _Identidade:                       # baseline (viola d < d', mas é só o ponto zero da escala)
    def fit(self, X): pass
    def phi(self, X): return X

class _RFF:                              # referência: random Fourier features, d' = 64
    def fit(self, X):
        g = torch.Generator().manual_seed(0)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        d2 = torch.cdist(Xs[:300], Xs[:300]) ** 2
        sigma = math.sqrt(d2[d2 > 0].median().item() / 2)
        self.W = torch.randn(X.shape[1], DIM_MAX, generator=g) / sigma
        self.b = torch.rand(DIM_MAX, generator=g) * 2 * math.pi
    def phi(self, X):
        return math.sqrt(2 / DIM_MAX) * torch.cos(((X - self.mu) / self.sd) @ self.W + self.b)


def _mediana(cls, gerador, checar=True):
    vals = [rodar(cls(), gerador, sem, checar) for sem in SEMENTES]
    return float(torch.tensor(vals).median())


def avaliar():
    print(f"{'tarefa':<12}{'baseline':>10}{'referência':>12}{'você':>8}{'s_t':>7}")
    s = []
    for nome, gen in TAREFAS.items():
        b = _mediana(_Identidade, gen, checar=False)
        r = max(_mediana(_RFF, gen, checar=False), b + 1e-3)
        try:
            m = _mediana(Submissao, gen); erro = ""
        except AssertionError as e:
            m, erro = b, f"   <- {e}"
        st = min(max((m - b) / (r - b), 0.0), 1.25); s.append(st)
        print(f"{nome:<12}{b:>10.3f}{r:>12.3f}{m:>8.3f}{st:>7.2f}{erro}")
    S = 100 * (0.7 * sum(s) / len(s) + 0.3 * min(s))
    print(f"\nESCORE S = {S:.1f}   (0 = baseline, 100 = referência, até 125 com bônus)")
    return S


if __name__ == "__main__":
    avaliar()