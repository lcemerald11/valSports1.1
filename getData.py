from bs4 import BeautifulSoup
import requests
import vlrdevapi as vlr
import json

def search_teams(team_query):
    url = f"https://www.vlr.gg/search/?q={team_query}&type=all"
    headers = {
        "User-Agent": "Mozilla/5.0"  # Helps avoid being blocked
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    teams = []

    # Search results are under divs with class 'search-item-team'
    team_id = 0
    low_id = 99999
    for team_div in soup.select(".wf-module-item.search-item"):
        name_tag = team_div.select_one(".search-item-title")  # Team name
        link_tag = team_div["href"]
        img = team_div.find("img")["src"]
        
        # print(link_tag[15:], link_tag.find("/idx"))

        if name_tag and "team" in link_tag:
            team_id = int(link_tag[15:link_tag.find("/idx")])
            # team_id = int(link_tag[15:link_tag.find("/")])
            name = name_tag.text.strip()
            link = f"https://www.vlr.gg" + team_div["href"]
            if ("//" in img):
                img = "https:" + img
            else:
                img = "https://www.vlr.gg" + img
            if (low_id >= team_id) and ("inactive" not in name):
                low_id = team_id
                teams.insert(0,[name, link, img, team_id])
            else:
                teams.append([name, link, img, team_id])
            # teams.append([name, link, img])
    return teams

def getTeamData(team_id):

    team = vlr.team(team_id=team_id)
    history = team.completed_matches()
    print(f"Completed matches for team {history.team_id}:")
    teamName = (vlr.team.info(team_id=team_id).name)
    print(teamName)
    currTeam = vlr.team(team_id=team_id)
    info = vlr.team.info(team_id=team_id)
    roster = team.roster()

    playerCurr = []
    for p in roster.players:
        playerCurr.append(p.ign)
    data = {
        "tag": info.tag,
        "id": info.id,
        "country": info.country,
        "winRate": 0.0,
        "currRoster": playerCurr
    }

    winNum = 0
    lossNum = 0
    for m in history.matches[:10]:
        currMatch = {}

        currMatch["id"] = m.match_id
        currMatch["opp"] = m.opponent.name if m.opponent else "TBD"
        currMatch["win"] = m.is_win

        if m.is_win:
            winNum = winNum + 1
        else:
            lossNum = lossNum + 1
        score = {
            teamName: m.team_score, 
            "opp": m.opponent_score,
        }
        currMatch["Score"] = score

        seriesStats = vlr.series(series_id=m.match_id)

        currMatch["Num Games"] = seriesStats.info().best_of

        vetoList = []
        pickList = []

        info = seriesStats.info()
        for v in info.veto:
            if v.team == data["tag"]:
                if v.veto_type == "pick":
                    pickList.append(v.map_name)
                elif v.veto_type == "ban":
                    vetoList.append(v.map_name)
                
        currMatch["Team P/B"] = {
            "vetoList": vetoList,
            "pickList": pickList
        }
        # print(currMatch["Team P/B"])
        mapStats = {}
        for x in range(1,currMatch["Num Games"] + 1):
            currMap = info.games[x-1]
            mapScore = [(info.team1.name,currMap.team1_score), (info.team2.name, currMap.team2_score)]
            if currMap.team1_score is not None:
                print(mapScore)
                totalStats = {
                    "score" : mapScore
                }
                playerStats = seriesStats.players()
                for team in [playerStats.team1, playerStats.team2]:
                    teamStats = {}
                    # print(f"--- {team.team_name} ---")
                    for player in team.players:
                        # print(player)
                        s = player.stats
                        playerData = {
                            "name": player.name,
                            "agents": player.agents,
                            "country": player.country
                        }
                        for side in s:
                            sideData = side[1]
                            rating = sideData.rating
                            acs = sideData.acs
                            kills = sideData.kills
                            deaths = sideData.deaths
                            assists = sideData.assists
                            kd_diff = sideData.kd_diff
                            kast = sideData.kast
                            adr = sideData.adr
                            firstKDdiff = sideData.fk_fd_diff

                            currSide = {
                                # "side": side[0]
                                "rating": rating,
                                "acs": acs,
                                "kills": kills,
                                "deaths": deaths,
                                "assists": assists,
                                "kd_diff": kd_diff,
                                "kast": kast,
                                "adr": adr,
                                "firstKDdiff": firstKDdiff
                            }

                            playerData[side[0]] = currSide
                        teamStats[f"{player.name}"] = playerData
                    totalStats[team.team_name] = teamStats
                game = info.games[x-1]
                # print(game)
                mapStats[f"{game.map_name}"] = totalStats
        currMatch["Each Map"] = mapStats
        data[f"Match {winNum+lossNum}"] = currMatch
    winRate = winNum/(winNum+lossNum)
    data["winRate"] = winRate

    return(data,teamName)

# getTeams = ["Loud","100 Thieves", "Cloud9", "ENVY", "Evil Geniuses", "Furia", "G2 Esports", "KRÜ Esports", "LEVIATÁN", "MIBR", "NRG", "Sentinels"]
# getTeams = ["100 Thieves", "Loud", "NRG", "JDG", "EDG" , "Xi Lai Gaming", "Karmine Corp", "FUT Esports", "Team Vitality", "Global Esports", "Nongshim Redforce", "T1"]
getTeams = ["Team Liquid", "G2 Esports", "Tyloo", "Paper Rex"]

for lmao in getTeams:
    team = search_teams(lmao)
    print(team[0][0])
    id = team[0][3]
    dataSaved = getTeamData(id)

    finalJson = json.dumps(dataSaved[0], indent=1)

    teamName = dataSaved[1] + ".json"

    with open(teamName, "a") as file:
        file.write(str(finalJson))
        file.close



