# main.py
from __future__ import annotations

import pandas as pd
import random
import uvicorn
import json
from uuid import uuid4
from io import StringIO
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Annotated, List, Dict
from fastapi import Form

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

results_cache: Dict[str, Dict] = {}


@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/assign", response_class=HTMLResponse)
async def assign_teams(
    request: Request, file: UploadFile = File(...), team_count: int = Form(...)
):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode("utf-8-sig")), index_col=False)
    df.reset_index(drop=True, inplace=True)

    participants = df.to_dict(orient="records")
    participants = sorted(participants, key=lambda x: x['이름'])

    teams = compute_teams(participants, team_count)

    token = str(uuid4())
    results_cache[token] = {
        "participants": participants,
        "teams": teams,
        "team_count": team_count,
    }
    return RedirectResponse(url=f"/result/{token}", status_code=303)


@app.post("/reassign", response_class=HTMLResponse)
async def reassign_teams(request: Request):
    form = await request.form()
    team_count = int(form["team_count"])
    player = form.getlist("참가자")

    rows = []
    i = 0
    while f"row_{i}" in form:
        row_data = json.loads(form[f"row_{i}"])
        row_data["참가여부"] = "true" if str(i) in player else "false"
        rows.append(row_data)
        i += 1

    teams = compute_teams(rows, team_count)

    token = str(uuid4())
    results_cache[token] = {
        "participants": sorted(rows, key=lambda x: x["이름"]),
        "teams": teams,
        "team_count": team_count,
    }
    return RedirectResponse(url=f"/result/{token}", status_code=303)

@app.get("/result/{token}", response_class=HTMLResponse)
async def show_result(request: Request, token: str):
    data = results_cache.get(token)
    if not data:                          # 잘못된 토큰 처리
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, **data}
    )


def compute_teams(
    participants: List[Dict], team_count: int
) -> List[List[Dict]]:
    df = pd.DataFrame(participants)
    df = df[df["참가여부"].astype(str).str.lower() == "true"].copy()
    df["분리그룹"] = df["분리그룹"].fillna('').astype(str)

    groups = []
    visited = set()
    for _, row in df.iterrows():
        g = row["분리그룹"]
        if g and g not in visited:
            visited.add(g)
            group_members = df[df["분리그룹"] == g].to_dict(orient="records")
            groups.append(group_members)
        elif not g:
            groups.append([row.to_dict()])

    teams = [[] for _ in range(team_count)]
    for group in groups:
        lengths = [len(team) for team in teams]
        min_len = min(lengths)
        candidate_indices = [i for i, l in enumerate(lengths) if l == min_len]
        random.shuffle(candidate_indices)
        for idx in candidate_indices:
            teams[idx].extend(group)
            break
        
    results = []
    for team in teams:
        male = sum(1 for p in team if p["성별"] == "남")
        female = sum(1 for p in team if p["성별"] == "여")
        total = len(team)
        avg_age = round(sum(int(p["나이"]) for p in team) / total, 1) if total else 0.0
        results.append({
            "members": team,
            "male": male,
            "female": female,
            "total": total,
            "avg_age": avg_age,
        })
    return results
    

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)