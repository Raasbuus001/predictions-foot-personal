"""
Prédiction de matchs de Premier League (version 1)
--------------------------------------------------
Modèle de Poisson avec force d'attaque / de défense par équipe, avantage du
terrain et pondération par ancienneté des matchs. Il estime :
  - le nombre de buts de chaque équipe (donc score, 1X2, over/under, BTTS)
  - les corners de chaque équipe
  - les cartons jaunes de chaque équipe

Usage :
    python predict.py "Arsenal" "Chelsea"
    python predict.py --backtest
    python predict.py --equipes
"""
import sys
import glob
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

DATA_GLOB = "data/E0_*.csv"
DEMI_VIE_JOURS = 365   # un match vieux d'un an compte deux fois moins
REGULARISATION = 1.0   # évite les valeurs extrêmes pour les équipes avec peu de matchs

CIBLES = {
    "buts": ("FTHG", "FTAG"),
    "corners": ("HC", "AC"),
    "jaunes": ("HY", "AY"),
    "rouges": ("HR", "AR"),
    "fautes": ("HF", "AF"),
    "mt": ("HTHG", "HTAG"),
    "tirs": ("HS", "AS"),
    "cadres": ("HST", "AST"),
}


def charger_donnees():
    frames = [pd.read_csv(f, encoding="utf-8-sig", encoding_errors="replace", on_bad_lines="skip") for f in sorted(glob.glob(DATA_GLOB))]
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    return df.sort_values("Date").reset_index(drop=True)


def ajuster(df, col_dom, col_ext, date_ref):
    """Ajuste un modèle de Poisson : log(mu) = intercept + attaque_equipe - defense_adverse (+ avantage domicile)."""
    d = df.dropna(subset=[col_dom, col_ext])
    equipes = sorted(set(d["HomeTeam"]) | set(d["AwayTeam"]))
    idx = {t: i for i, t in enumerate(equipes)}
    n = len(equipes)
    ages = (date_ref - d["Date"]).dt.days.values
    poids = 0.5 ** (ages / DEMI_VIE_JOURS)
    poids = np.concatenate([poids, poids])

    dom = d["HomeTeam"].map(idx).values
    ext = d["AwayTeam"].map(idx).values
    y = np.concatenate([d[col_dom].values, d[col_ext].values]).astype(float)
    att = np.concatenate([dom, ext])
    deff = np.concatenate([ext, dom])
    est_dom = np.concatenate([np.ones(len(d)), np.zeros(len(d))])

    def nll(p):
        mu_log = p[0] + p[1 + att] - p[1 + n + deff] + p[1 + 2 * n] * est_dom
        mu = np.exp(mu_log)
        ll = poids * (y * mu_log - mu)
        pen = REGULARISATION * (np.sum(p[1:1 + n] ** 2) + np.sum(p[1 + n:1 + 2 * n] ** 2))
        grad_mu = poids * (y - mu)
        g = np.zeros_like(p)
        g[0] = grad_mu.sum()
        np.add.at(g, 1 + att, grad_mu)
        np.add.at(g, 1 + n + deff, -grad_mu)
        g[1 + 2 * n] = (grad_mu * est_dom).sum()
        g[1:1 + n] -= 2 * REGULARISATION * p[1:1 + n]
        g[1 + n:1 + 2 * n] -= 2 * REGULARISATION * p[1 + n:1 + 2 * n]
        return -(ll.sum() - pen), -g

    p0 = np.zeros(2 * n + 2)
    p0[0] = np.log(y.mean())
    res = minimize(nll, p0, jac=True, method="L-BFGS-B")
    p = res.x
    return {"equipes": idx, "n": n, "intercept": p[0], "att": p[1:1 + n],
            "def": p[1 + n:1 + 2 * n], "dom": p[1 + 2 * n]}


def attendu(m, dom, ext):
    i, j = m["equipes"][dom], m["equipes"][ext]
    mu_dom = np.exp(m["intercept"] + m["att"][i] - m["def"][j] + m["dom"])
    mu_ext = np.exp(m["intercept"] + m["att"][j] - m["def"][i])
    return mu_dom, mu_ext


def modeles(df, date_ref):
    train = df[df["Date"] < date_ref]
    return {nom: ajuster(train, c1, c2, date_ref) for nom, (c1, c2) in CIBLES.items()}


def matrice_scores(mu_dom, mu_ext, max_buts=10):
    a = poisson.pmf(np.arange(max_buts + 1), mu_dom)
    b = poisson.pmf(np.arange(max_buts + 1), mu_ext)
    M = np.outer(a, b)
    # Petite correction de type Dixon-Coles sur les scores faibles (0-0, 1-0, 0-1, 1-1)
    rho = -0.05
    M[0, 0] *= 1 - mu_dom * mu_ext * rho
    M[0, 1] *= 1 + mu_dom * rho
    M[1, 0] *= 1 + mu_ext * rho
    M[1, 1] *= 1 - rho
    return M / M.sum()


def resume_buts(M):
    tri = np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()
    total = np.add.outer(np.arange(M.shape[0]), np.arange(M.shape[1]))
    over = {s: M[total > s].sum() for s in (0.5, 1.5, 2.5, 3.5)}
    btts = M[1:, 1:].sum()
    return {"1": tri[0], "N": tri[1], "2": tri[2], "over": over, "btts": btts}


def predire(df, dom, ext, date_ref=None):
    date_ref = date_ref or (df["Date"].max() + pd.Timedelta(days=1))
    mods = modeles(df, date_ref)
    for e in (dom, ext):
        if e not in mods["buts"]["equipes"]:
            raise SystemExit(f"Équipe inconnue : {e}. Lance `python predict.py --equipes`.")
    mu = {nom: attendu(m, dom, ext) for nom, m in mods.items()}
    M = matrice_scores(*mu["buts"])
    r = resume_buts(M)
    scores = sorted(((M[i, j], i, j) for i in range(7) for j in range(7)), reverse=True)[:5]

    print(f"\n=== {dom} (domicile) vs {ext} ===")
    print(f"Buts attendus   : {mu['buts'][0]:.2f} - {mu['buts'][1]:.2f}  (total {sum(mu['buts']):.2f})")
    print(f"Victoire {dom}: {r['1']:.0%} | Nul: {r['N']:.0%} | Victoire {ext}: {r['2']:.0%}")
    print("Scores les plus probables :", ", ".join(f"{i}-{j} ({p:.0%})" for p, i, j in scores))
    print("Plus de 1.5 buts : {:.0%} | Plus de 2.5 : {:.0%} | Plus de 3.5 : {:.0%}".format(
        r["over"][1.5], r["over"][2.5], r["over"][3.5]))
    print(f"Les deux équipes marquent : {r['btts']:.0%}")
    c = mu["corners"]
    print(f"Corners attendus: {c[0]:.1f} - {c[1]:.1f}  (total {sum(c):.1f})")
    j = mu["jaunes"]
    print(f"Cartons jaunes  : {j[0]:.1f} - {j[1]:.1f}  (total {sum(j):.1f})")
    r_ = mu["rouges"]
    print(f"Cartons rouges  : {r_[0]:.2f} - {r_[1]:.2f}  (au moins un : {1 - np.exp(-sum(r_)):.0%})")
    f_ = mu["fautes"]
    print(f"Fautes          : {f_[0]:.1f} - {f_[1]:.1f}  (total {sum(f_):.1f})")
    print("Penalties       : non disponible (absent des données)\n")


def backtest(df, saison_debut="2025-08-01", saison_fin="2026-06-30", pas_jours=28):
    """Simule des prédictions sur la saison passée avec seulement les données connues avant chaque match."""
    test = df[(df["Date"] >= saison_debut) & (df["Date"] <= saison_fin)].copy()
    debuts = pd.date_range(saison_debut, saison_fin, freq=f"{pas_jours}D")
    lignes = []
    for k, d0 in enumerate(debuts):
        d1 = debuts[k + 1] if k + 1 < len(debuts) else pd.Timestamp(saison_fin) + pd.Timedelta(days=1)
        bloc = test[(test["Date"] >= d0) & (test["Date"] < d1)]
        if bloc.empty:
            continue
        mods = modeles(df, d0)
        for _, row in bloc.iterrows():
            h, a = row["HomeTeam"], row["AwayTeam"]
            if h not in mods["buts"]["equipes"] or a not in mods["buts"]["equipes"]:
                continue
            mb = attendu(mods["buts"], h, a)
            M = matrice_scores(*mb)
            r = resume_buts(M)
            mc = attendu(mods["corners"], h, a)
            lignes.append({
                "p1": r["1"], "pN": r["N"], "p2": r["2"], "over25": r["over"][2.5],
                "btts": r["btts"], "mu_buts": sum(mb), "mu_corners": sum(mc),
                "res": row["FTR"], "buts": row["FTHG"] + row["FTAG"],
                "corners": row["HC"] + row["AC"] if pd.notna(row["HC"]) else np.nan,
                "bttsr": int(row["FTHG"] > 0 and row["FTAG"] > 0),
                "B365H": row.get("B365H"), "B365D": row.get("B365D"), "B365A": row.get("B365A"),
            })
    b = pd.DataFrame(lignes)
    print(f"\nBacktest sur {len(b)} matchs ({saison_debut} -> {saison_fin})")

    probs = b[["p1", "pN", "p2"]].values
    vrai = b["res"].map({"H": 0, "D": 1, "A": 2}).values
    pred = probs.argmax(1)
    ll = -np.log(probs[np.arange(len(b)), vrai]).mean()
    ll_base = -np.log(np.array([0.45, 0.25, 0.30])[vrai]).mean()
    print(f"Résultat 1X2 : {np.mean(pred == vrai):.1%} de bons pronostics (log-loss {ll:.3f}, "
          f"repère 'toujours 45/25/30' : {ll_base:.3f})")

    bk = b.dropna(subset=["B365H", "B365D", "B365A"])
    if len(bk):
        inv = 1 / bk[["B365H", "B365D", "B365A"]].values
        pb = inv / inv.sum(1, keepdims=True)
        vb = bk["res"].map({"H": 0, "D": 1, "A": 2}).values
        print(f"Bookmaker (Bet365) : {np.mean(pb.argmax(1) == vb):.1%} de bons pronostics "
              f"(log-loss {-np.log(pb[np.arange(len(bk)), vb]).mean():.3f})")

    print(f"Buts totaux : erreur moyenne {np.abs(b['mu_buts'] - b['buts']).mean():.2f} "
          f"(en prédisant toujours la moyenne : {np.abs(b['buts'].mean() - b['buts']).mean():.2f})")
    o = ((b["over25"] > 0.5) == (b["buts"] > 2.5)).mean()
    print(f"Plus/moins de 2.5 buts : {o:.1%} de bons pronostics "
          f"(taux réel de plus de 2.5 : {(b['buts'] > 2.5).mean():.1%})")
    print(f"Les deux marquent : {((b['btts'] > 0.5) == (b['bttsr'] == 1)).mean():.1%} de bons pronostics")
    c = b.dropna(subset=["corners"])
    print(f"Corners totaux : erreur moyenne {np.abs(c['mu_corners'] - c['corners']).mean():.2f} "
          f"(en prédisant toujours la moyenne : {np.abs(c['corners'].mean() - c['corners']).mean():.2f})")


if __name__ == "__main__":
    donnees = charger_donnees()
    args = sys.argv[1:]
    if args == ["--backtest"]:
        backtest(donnees)
    elif args == ["--equipes"]:
        print("\n".join(sorted(set(donnees["HomeTeam"]) | set(donnees["AwayTeam"]))))
    elif len(args) == 2:
        predire(donnees, args[0], args[1])
    else:
        print(__doc__)
