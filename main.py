# main.py
from __future__ import annotations

import pandas as pd
import random
import uvicorn
import json
from uuid import uuid4
from io import StringIO
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Annotated, List, Dict
from fastapi import Form
from collections import defaultdict

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

results_cache: Dict[str, Dict] = {}


@app.get("/logo.png", response_class=FileResponse)
async def logo():
    return FileResponse("logo.png", media_type="image/png")


@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/assign", response_class=HTMLResponse)
async def assign_teams(
    request: Request, file: UploadFile = File(...), team_count: int = Form(...)
):
    contents = await file.read()
    try:
        df = pd.read_csv(StringIO(contents.decode("utf-8-sig")), index_col=False)
        df.reset_index(drop=True, inplace=True)
        participants = df.to_dict(orient="records")
    except Exception:
        # CSV 파싱 실패 시 기본값으로 처리
        participants = []

    participants = sorted(participants, key=lambda x: x.get('이름', ''))

    teams = compute_teams(participants, team_count)

    token = str(uuid4())
    results_cache[token] = {
        "participants": participants,
        "teams": teams,
        "team_count": team_count,
        "token": token
    }
    return RedirectResponse(url=f"/result/{token}", status_code=303)


@app.post("/api/reassign", response_class=JSONResponse)
async def api_reassign_teams(request: Request):
    form = await request.form()
    team_count = int(form.get("team_count", 1))
    player = form.getlist("참가자")

    rows = []
    i = 0
    while f"row_{i}" in form:
        row_data = json.loads(form[f"row_{i}"])
        row_data["참가여부"] = "true" if str(i) in player else "false"
        rows.append(row_data)
        i += 1
    
    teams = compute_teams(rows, team_count)
    
    return {"teams": teams, "team_count": team_count}


@app.get("/result/{token}", response_class=HTMLResponse)
async def show_result(request: Request, token: str):
    data = results_cache.get(token)
    if not data:
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, **data}
    )

@app.get("/edit/{token}", response_class=HTMLResponse)
async def edit_page(request: Request, token: str):
    data = results_cache.get(token)
    if not data:
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse("edit.html", {"request": request, **data})

@app.post("/update/{token}", response_class=RedirectResponse)
async def update_data(request: Request, token: str):
    if token not in results_cache:
        return RedirectResponse("/", status_code=302)

    form_data = await request.form()
    
    new_participants = []
    keys = [k.split('_')[0] for k in form_data.keys() if k.endswith('_0')]
    
    num_participants = len([k for k in form_data.keys() if k.startswith('이름_')])

    for i in range(num_participants):
        participant = {}
        for key in keys:
            form_key = f"{key}_{i}"
            if form_key in form_data:
                participant[key] = form_data[form_key]
        if participant.get("이름"): # 이름이 있는 경우에만 추가
            new_participants.append(participant)

    # 캐시 업데이트
    results_cache[token]["participants"] = sorted(new_participants, key=lambda x: x.get('이름', ''))
    
    # 수정된 명단으로 팀 재계산
    team_count = results_cache[token]["team_count"]
    teams = compute_teams(new_participants, team_count)
    results_cache[token]["teams"] = teams

    return RedirectResponse(url=f"/result/{token}", status_code=303)


def compute_teams(
    participants: List[Dict], team_count: int
) -> List[List[Dict]]:
    
    active_participants_df = pd.DataFrame(participants)
    active_participants_df = active_participants_df[active_participants_df["참가여부"].astype(str).str.lower() == "true"].copy()
    
    if active_participants_df.empty:
        return []

    active_participants_df["분리그룹"] = active_participants_df["분리그룹"].fillna('').astype(str)

    separated_participants_map = defaultdict(list)
    normal_participants = []

    for participant in active_participants_df.to_dict(orient="records"):
        separation_group = participant.get("분리그룹")
        if separation_group and str(separation_group).strip():
            separated_participants_map[separation_group].append(participant)
        else:
            normal_participants.append(participant)

    teams = [[] for _ in range(team_count)]
    team_indices = list(range(team_count))
    
    for group_name, members in separated_participants_map.items():
        random.shuffle(team_indices)
        for i, member in enumerate(members):
            target_team_index = team_indices[i % team_count]
            teams[target_team_index].append(member)

    random.shuffle(normal_participants)
    
    for participant in normal_participants:
        teams.sort(key=len)
        teams[0].append(participant)

    results = []
    for team in teams:
        team.sort(key=lambda p: p.get('이름', ''))
        
        male = sum(1 for p in team if p.get("성별") == "남")
        female = sum(1 for p in team if p.get("성별") == "여")
        total = len(team)
        results.append({
            "members": team,
            "male": male,
            "female": female,
            "total": total,
        })
    return results
    

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
