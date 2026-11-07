"""Génère index.html : ajuste les modèles de chaque championnat et les intègre dans la page.
Pour ajouter un championnat : mettre ses CSV dans data/ (ex. data/SP1_2025-26.csv) puis l'ajouter dans LIGUES."""
import json, os
import predict
import numpy as np
K_ELO = 0.00285  # log(buts attendus) par point Elo, mesuré sur la Premier League
# Niveau moyen des championnats (relevé ClubElo du 2026-11-07), utilisé si le téléchargement en direct échoue
ELO_MOYEN = {"E0": 1837.5, "E1": 1610.3, "SP1": 1763.8, "SP2": 1623, "F1": 1713.8, "F2": 1540,
             "D1": 1752, "D2": 1591, "I1": 1739, "I2": 1543}
if os.path.exists("elo_live.json"):
    ELO_MOYEN.update(json.load(open("elo_live.json")))

LIGUES = {  # code : (nom, motif des fichiers)
    "E0": "Premier League", "E1": "Championship",
    "SP1": "La Liga", "SP2": "La Liga 2",
    "F1": "Ligue 1", "F2": "Ligue 2",
    "D1": "Bundesliga", "D2": "2. Bundesliga",
    "I1": "Serie A", "I2": "Serie B",
}
A_VENIR = []

out = {}
for code, nom in LIGUES.items():
    predict.DATA_GLOB = f"data/{code}_*.csv"
    df = predict.charger_donnees()
    ref = df["Date"].max() + __import__("pandas").Timedelta(days=1)
    mods = predict.modeles(df, ref)
    teams = sorted(mods["buts"]["equipes"])
    m = {}
    for k, v in mods.items():
        order = [v["equipes"][t] for t in teams]
        m[k] = {"c": round(float(v["intercept"]), 4), "h": round(float(v["dom"]), 4),
                "a": [round(float(v["att"][i]), 4) for i in order],
                "d": [round(float(v["def"][i]), 4) for i in order]}
    recents = set(df[df["Date"] >= df["Date"].max() - __import__("pandas").Timedelta(days=120)]["HomeTeam"])
    act = [i for i, t in enumerate(teams) if t in recents]
    bm = mods["buts"]
    q = [float(bm["att"][bm["equipes"][teams[i]]] + bm["def"][bm["equipes"][teams[i]]]) for i in act]
    off = round(K_ELO * (ELO_MOYEN[code] - 1700) - float(np.mean(q)), 4)
    out[code] = {"nom": nom, "off": off, "teams": teams, "act": act, "m": m, "maj": str(df["Date"].max().date()), "n": len(df)}

html = open("template.html", encoding="utf-8").read()
html = html.replace("__DATA__", json.dumps(out, ensure_ascii=False)).replace("__AVENIR__", json.dumps(A_VENIR, ensure_ascii=False))
open("index.html", "w", encoding="utf-8").write(html)
print("OK", {k: len(v["teams"]) for k, v in out.items()})
