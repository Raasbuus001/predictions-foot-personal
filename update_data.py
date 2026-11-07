"""Met à jour les données : résultats (football-data.co.uk), niveau des championnats (ClubElo),
joueurs indisponibles de Premier League (Fantasy Premier League). Lancé chaque jour par GitHub."""
import csv, io, json, os, datetime, urllib.request

LIGUES = ["E0", "E1", "SP1", "SP2", "F1", "F2", "D1", "D2", "I1", "I2"]
PAYS = {"ENG": ("E0", "E1"), "ESP": ("SP1", "SP2"), "FRA": ("F1", "F2"), "GER": ("D1", "D2"), "ITA": ("I1", "I2")}
NOMS_FPL = {"Man Utd": "Man United", "Spurs": "Tottenham"}  # noms FPL -> noms des fichiers de résultats


def get(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (prediction-foot)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def resultats():
    os.makedirs("data", exist_ok=True)
    now = datetime.date.today()
    debut = now.year if now.month >= 7 else now.year - 1
    ok = ko = 0
    for y in range(debut - 3, debut + 1):
        code, label = f"{y % 100:02d}{(y + 1) % 100:02d}", f"{y}-{(y + 1) % 100:02d}"
        for lg in LIGUES:
            dest = f"data/{lg}_{label}.csv"
            try:
                raw = get(f"https://www.football-data.co.uk/mmz4281/{code}/{lg}.csv")
                if b"HomeTeam" not in raw[:2000]:
                    raise ValueError("fichier inattendu")
                open(dest, "wb").write(raw); ok += 1
            except Exception as e:
                ko += 1
                print(f"  ! {dest} : {e}" + (" (ancien fichier conservé)" if os.path.exists(dest) else ""))
    print(f"Résultats : {ok} fichiers téléchargés, {ko} échecs")


def elo():
    try:
        raw = get(f"http://api.clubelo.com/{datetime.date.today().isoformat()}").decode("utf-8", "replace")
        somme, n = {}, {}
        for r in csv.DictReader(io.StringIO(raw)):
            if r.get("Country") in PAYS and r.get("Level") in ("1", "2"):
                lg = PAYS[r["Country"]][int(r["Level"]) - 1]
                somme[lg] = somme.get(lg, 0) + float(r["Elo"]); n[lg] = n.get(lg, 0) + 1
        moy = {lg: round(somme[lg] / n[lg], 1) for lg in somme if n[lg] >= 10}
        if len(moy) == len(LIGUES):
            json.dump(moy, open("elo_live.json", "w")); print("Elo mis à jour :", moy)
        else:
            print("Elo : données incomplètes, valeurs précédentes conservées", moy)
    except Exception as e:
        print("Elo : échec, valeurs précédentes conservées :", e)


def blesses():
    try:
        d = json.loads(get("https://fantasy.premierleague.com/api/bootstrap-static/"))
        equipes = {t["id"]: NOMS_FPL.get(t["name"], t["name"]) for t in d["teams"]}
        out = {n: [] for n in equipes.values()}
        for p in d["elements"]:
            if p["status"] in ("i", "d", "s"):
                out[equipes[p["team"]]].append({"n": p["web_name"], "s": p["status"],
                                                "c": p.get("chance_of_playing_next_round"), "t": p.get("news", "")})
        for l in out.values():
            l.sort(key=lambda x: (x["s"] != "i", x["s"] != "s", x["n"]))
        json.dump({"maj": datetime.date.today().strftime("%d/%m/%Y"), "teams": out}, open("injuries.json", "w"), ensure_ascii=False)
        print("Blessés mis à jour :", sum(len(v) for v in out.values()), "joueurs signalés")
    except Exception as e:
        print("Blessés : échec, fichier précédent conservé :", e)


if __name__ == "__main__":
    resultats(); elo(); blesses()
