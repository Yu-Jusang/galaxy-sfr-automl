"""Build docs/index.html (GitHub Pages) from results/meta/*.json and results/status.json.

Layout follows the KCB work-log site: status strip, numbered sections (overview, environment, workflow,
timeline, results, downloads, reproduction). Run any time; the page reflects whatever has finished so far.
"""
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd  # noqa: E402

META = ROOT / "results" / "meta"
STATUS = ROOT / "results" / "status.json"
FINAL = ROOT / "results" / "final.json"
DOCS = ROOT / "docs"
R0 = "M01_lgbm_tw13_F5"
STAGE_KO = {"single": "단일 모델", "autogluon": "AutoGluon", "ensemble": "앙상블"}


def esc(s):
    return html.escape(str(s))


def table():
    rows = []
    for f in sorted(META.glob("*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        fd = pd.DataFrame(m["folds"])
        rows.append({"name": m["name"], "stage": m.get("stage", ""), "features": m.get("features", ""), "desc": m.get("desc", ""),
                     "cv": fd.rmse_all.mean(), "zero": fd.rmse_zero.mean(), "top10": fd.rmse_top10.mean(),
                     "log10": fd.log10rmse_pos.mean(), "min": m.get("seconds", 0) / 60, "f": fd.rmse_all.to_numpy()})
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows)
    if (t.name == R0).any():
        r0 = t.loc[t.name == R0, "f"].iloc[0]
        d = t.f.apply(lambda a: a - r0)
        t["diff"] = d.apply(lambda a: a.mean())
        t["se"] = d.apply(lambda a: a.std(ddof=1) / 5 ** 0.5)
        t["nb"] = d.apply(lambda a: int((a < 0).sum()))
    return t.sort_values("cv").reset_index(drop=True)


def results_html(t):
    if t.empty:
        return "<p class='muted'>아직 끝난 실험이 없습니다.</p>"
    out = ["<div class='tbl'><table><thead><tr><th>#</th><th>실험</th><th>종류</th><th>변수</th>"
           "<th class='n'>CV RMSE</th><th class='n'>R0 대비 Δ (SE)</th><th class='n'>이긴 fold</th>"
           "<th class='n'>SFR=0 RMSE</th><th class='n'>상위10% RMSE</th><th class='n'>분</th></tr></thead><tbody>"]
    for i, r in t.iterrows():
        cls = " class='r0'" if r["name"] == R0 else (" class='best'" if i == 0 else "")
        dd = "" if r["name"] == R0 or pd.isna(r.get("diff")) else f"{r['diff']:+.3f} ({r['se']:.3f})"
        nb = "" if r["name"] == R0 or pd.isna(r.get("nb")) else f"{int(r['nb'])}/5"
        out.append(f"<tr{cls}><td>{i + 1}</td><td><b>{esc(r['name'])}</b><div class='muted'>{esc(r['desc'])}</div></td>"
                   f"<td>{STAGE_KO.get(r['stage'], r['stage'])}</td><td>{esc(r['features'])}</td>"
                   f"<td class='n'><b>{r['cv']:.3f}</b></td><td class='n'>{dd}</td><td class='n'>{nb}</td>"
                   f"<td class='n'>{r['zero']:.2f}</td><td class='n'>{r['top10']:.1f}</td><td class='n'>{r['min']:.0f}</td></tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def main():
    DOCS.mkdir(exist_ok=True)
    t = table()
    st = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {"log": []}
    fin = json.loads(FINAL.read_text(encoding="utf-8")) if FINAL.exists() else None
    n_single = int((t.stage == "single").sum()) if not t.empty else 0
    n_ag = int((t.stage == "autogluon").sum()) if not t.empty else 0
    n_ens = int((t.stage == "ensemble").sum()) if not t.empty else 0
    best = t.iloc[0] if not t.empty else None
    if fin:
        strip = (f"<b>최종:</b> {esc(fin['label'])} · 검증 RMSE <b>{fin['val_rmse']:.3f}</b> "
                 f"(원래 최종 모델 {fin['r0_val_rmse']:.3f}, 양식 HGB 9.525) · R² {fin['val_r2']:.3f} · "
                 f"<a href='downloads/galaxy_sfr_automl_report.pdf'>보고서 PDF</a>")
    elif best is not None:
        strip = (f"<b>진행 중:</b> {esc(st.get('message', ''))} · 지금까지 CV 1위 <b>{esc(best['name'])}</b> "
                 f"(CV RMSE {best['cv']:.3f}; 원래 최종 모델 7.401)")
    else:
        strip = "<b>준비 중</b>"
    log = st.get("log", [])
    # timeline: keep the last message per experiment
    seen, items = {}, []
    for e in log:
        key = e["msg"].split(":")[0]
        seen[key] = e
    for e in sorted(seen.values(), key=lambda e: e["t"]):
        items.append(f"<li><span class='t'>{esc(e['t'][5:16])}</span><span class='chip'>{esc(STAGE_KO.get(e['stage'], e['stage']))}</span>"
                     f"{esc(e['msg'])}</li>")
    page = TEMPLATE.format(
        strip=strip, updated=esc(st.get("updated", "")), n_single=n_single, n_ag=n_ag, n_ens=n_ens,
        results=results_html(t), timeline="".join(items[-60:]) or "<li class='muted'>기록 없음</li>",
        final_block=final_html(fin))
    (DOCS / "index.html").write_text(page, encoding="utf-8")
    print("docs/index.html written;", len(t), "experiments")


def final_html(fin):
    if not fin:
        return "<p class='muted'>모든 실험이 끝나면 CV 1위 후보를 고르고 검증셋을 한 번 평가해 여기에 적습니다.</p>"
    rows = "".join(f"<tr><td>{esc(r['label'])}</td><td class='n'>{r['cv']:.3f}</td><td class='n'>{r['val_rmse']:.3f}</td>"
                   f"<td class='n'>{r['val_mae']:.3f}</td><td class='n'>{r['val_r2']:.3f}</td></tr>" for r in fin["table"])
    return (f"<p>{esc(fin['summary'])}</p><div class='tbl'><table><thead><tr><th>모델</th><th class='n'>CV RMSE</th>"
            f"<th class='n'>검증 RMSE</th><th class='n'>검증 MAE</th><th class='n'>R²</th></tr></thead><tbody>{rows}</tbody></table></div>"
            f"<p class='muted'>최종 − 원래 최종 모델: Δ {fin['delta']:+.3f}, 95% bootstrap CI [{fin['ci'][0]:+.3f}, {fin['ci'][1]:+.3f}]</p>")


TEMPLATE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>은하 SFR AutoML 실험</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css">
<style>
:root{{--bg:#f7f8fa;--card:#fff;--ink:#16181d;--mute:#6b7280;--line:#e5e7eb;--acc:#2563eb;--acc2:#e8f0fe;--good:#0f7b4a;--goodbg:#e7f6ee}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0f1115;--card:#171a21;--ink:#e6e8ec;--mute:#9aa3b2;--line:#2a2f3a;--acc:#7aa2ff;--acc2:#1c2540;--good:#5fd39a;--goodbg:#14281f}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:"Pretendard Variable",Pretendard,system-ui,sans-serif;line-height:1.65;font-size:15px}}
.wrap{{max-width:1080px;margin:0 auto;padding:28px 16px 80px}}
.eyebrow{{color:var(--acc);font-weight:700;font-size:13px;letter-spacing:.04em}}
h1{{font-size:30px;margin:4px 0 6px}}.lead{{color:var(--mute);margin:0 0 18px}}
.strip{{background:var(--acc2);border:1px solid var(--line);border-radius:12px;padding:12px 16px;margin:14px 0}}
.strip a{{color:var(--acc)}}.upd{{color:var(--mute);font-size:13px}}
.nav{{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 8px}}.nav a{{text-decoration:none;color:var(--ink);border:1px solid var(--line);background:var(--card);border-radius:999px;padding:4px 12px;font-size:13px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px 22px;margin:16px 0}}
h2{{margin:0 0 12px;font-size:20px;display:flex;gap:10px;align-items:baseline}}h2 .no{{color:var(--acc);font-size:14px;font-weight:800}}
dl{{display:grid;grid-template-columns:120px 1fr;gap:8px 16px;margin:0}}dt{{font-weight:700}}dd{{margin:0}}
.flow{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}}.flow div{{border:1px solid var(--line);border-radius:10px;padding:10px 12px}}.flow b{{display:block}}
.tl{{list-style:none;padding:0;margin:0}}.tl li{{padding:6px 0;border-bottom:1px dashed var(--line)}}.t{{font-variant-numeric:tabular-nums;color:var(--mute);margin-right:10px}}
.chip{{display:inline-block;font-size:12px;border-radius:999px;padding:0 8px;margin-right:8px;background:var(--acc2);color:var(--acc)}}
.tbl{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13.5px}}th,td{{padding:7px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{color:var(--mute);font-weight:600;white-space:nowrap}}.n{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
tr.best td{{background:var(--goodbg)}}tr.r0 td{{background:var(--acc2)}}.muted{{color:var(--mute);font-size:12.5px}}
.dl a{{display:inline-block;margin:4px 8px 4px 0;padding:8px 14px;border-radius:10px;border:1px solid var(--line);text-decoration:none;color:var(--ink);background:var(--bg)}}
code{{font-family:ui-monospace,Consolas,monospace;font-size:13px;background:var(--bg);padding:1px 5px;border-radius:5px}}
pre{{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:12px;overflow-x:auto}}
@media (max-width:640px){{dl{{grid-template-columns:1fr}}h1{{font-size:24px}}}}
</style></head><body><div class="wrap">
<div class="eyebrow">SNU DataHub · 천문 데이터 · 작업 기록</div>
<h1>은하 SFR 예측 — AutoML 총동원 실험</h1>
<p class="lead">원래 최종 모델(LightGBM + Tweedie, CV RMSE 7.401)을 더 다양한 모델·AutoGluon·앙상블로 이길 수 있는지 확인한 기록입니다.</p>
<div class="strip">{strip}<div class="upd">마지막 갱신 {updated} KST</div></div>
<nav class="nav"><a href="#s1">01 개요</a><a href="#s2">02 계산 환경</a><a href="#s3">03 워크플로</a><a href="#s4">04 진행 기록</a><a href="#s5">05 결과</a><a href="#s6">06 다운로드</a><a href="#s7">07 재현 방법</a></nav>

<section id="s1"><h2><span class="no">01</span>개요</h2><dl>
<dt>과제</dt><dd>시뮬레이션 은하 208,437개의 질량·성장·환경으로 별 생성률 <code>sfr_linear</code>를 예측하는 회귀. SFR = 0이 8.66%, 최댓값 2,145의 두꺼운 꼬리.</dd>
<dt>평가</dt><dd>양식 고정 80/20 분할의 검증셋(41,688개) 선형 RMSE. 낮을수록 좋다.</dd>
<dt>출발점</dt><dd>원래 최종 모델 R0 = 무튜닝 LightGBM, Tweedie p = 1.3, 변수 25개(F5). CV 7.401, 검증 7.619.</dd>
<dt>이번 목표</dt><dd>단일 모델 총동원 → AutoGluon(전처리 7가지) → 앙상블로 RMSE를 최대한 낮춘다.</dd>
<dt>선택 원칙</dt><dd>모든 비교는 학습 80% 안의 같은 5-fold CV. <b>최종 후보 = CV RMSE가 가장 낮은 것</b>. 검증셋은 선택이 끝난 뒤 한 번만 평가한다.</dd>
<dt>공개 범위</dt><dd>코드·결과 요약·보고서만 올린다. 원본 CSV와 개인정보는 올리지 않는다(데이터: Fukushima & Kim, SNU DataHub dataset 42, CC BY 4.0).</dd>
</dl></section>

<section id="s2"><h2><span class="no">02</span>계산 환경</h2><dl>
<dt>기기</dt><dd>로컬 Windows 노트북 · Intel i7-1360P(16스레드) · RAM 16 GB · GPU 없음</dd>
<dt>패키지</dt><dd>Python 3.11, AutoGluon 1.4.0, LightGBM 4.6.0, XGBoost 3.0.5, CatBoost 1.2.10, scikit-learn 1.7.2, PyTorch 2.7.1 (CPU)</dd>
<dt>시간</dt><dd>AutoGluon 설정당 최대 1.5시간, 전체 약 12–16시간</dd>
</dl></section>

<section id="s3"><h2><span class="no">03</span>워크플로</h2><div class="flow">
<div><b>0. 재현</b>원래 최종 모델 CV 7.4014 재현</div>
<div><b>1. 단일 모델 {n_single}개</b>LightGBM·XGBoost·CatBoost(Tweedie/Poisson/RMSE), HGB, RF, ExtraTrees, MLP</div>
<div><b>2. AutoGluon {n_ag}개</b>best_quality, 변수 세트·목표 변환·Tweedie 추가를 바꿔 가며</div>
<div><b>3. 앙상블 {n_ens}개</b>OOF 예측으로 가중 평균, 가중치도 CV 안에서 학습</div>
<div><b>4. 최종</b>CV 1위 선택 → 검증셋 1회 평가</div>
</div></section>

<section id="s4"><h2><span class="no">04</span>진행 기록</h2><ul class="tl">{timeline}</ul></section>

<section id="s5"><h2><span class="no">05</span>결과</h2>
<h3 style="margin:4px 0 8px">최종 선택과 검증셋</h3>{final_block}
<h3 style="margin:18px 0 8px">전체 실험 (CV RMSE 순)</h3>
<p class="muted">CV RMSE = 학습 80% 안 5-fold 평균. Δ = 같은 fold끼리 뺀 차이의 평균(음수면 R0보다 좋음), SE = 그 표준오차. 파란 줄은 R0, 초록 줄은 현재 1위.</p>
{results}</section>

<section id="s6"><h2><span class="no">06</span>다운로드</h2><div class="dl">
<a href="downloads/galaxy_sfr_automl_report.pdf">보고서 PDF</a><a href="downloads/galaxy_sfr_automl_code.zip">코드 전체 (.zip)</a>
<a href="https://github.com/Yu-Jusang/galaxy-sfr-automl">GitHub 저장소</a></div>
<p class="muted">보고서와 zip은 실험이 끝난 뒤 올라갑니다.</p></section>

<section id="s7"><h2><span class="no">07</span>재현 방법</h2>
<pre>conda create -n sfr-ag python=3.11 -y
pip install "autogluon.tabular[all]==1.4.0"
# DataHub에서 받은 CSV를 data/ 에 둔다
bash scripts/run_all.sh          # 단일 모델 → AutoGluon
python scripts/a04_ensemble.py   # 앙상블과 최종 선택, 검증셋 1회 평가</pre></section>
</div></body></html>"""

if __name__ == "__main__":
    main()
