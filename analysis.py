"""
analysis.py  —  연령에 따른 음악 소비 패턴 분석
김정연 | 데이터마이닝 2026-1학기

데이터:
  - SpotifyFeatures.csv  (232K 트랙, 장르별 오디오 피처)
  - dataset.csv          (114K 트랙, 세부 장르 보강)
  - songs_normalize.csv  (Top Hits 2000-2019, 연도 정보)

실행: python analysis.py
결과: results/ 폴더에 PNG 저장
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# ── 한글 폰트 ───────────────────────────────────────────────
_nanum = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if os.path.exists(_nanum):
    fm.fontManager.addfont(_nanum)
    matplotlib.rcParams['font.family'] = fm.FontProperties(fname=_nanum).get_name()
matplotlib.rcParams['axes.unicode_minus'] = False

PALETTE = ["#4A6FA5","#E07B54","#6AAB9C","#C75E7E","#8B7BB5","#D4B483"]
NAVY    = "#2B3A67"

# ── 1. 로드 ─────────────────────────────────────────────────
def load(path="data/SpotifyFeatures_merged.csv"):
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    df = df[df['popularity'] > 0]
    df = df[(df['duration_ms'] >= 30_000) & (df['duration_ms'] <= 900_000)]
    df['duration_min'] = df['duration_ms'] / 60_000

    # 세대 프록시
    age_map = {
        "Classical":"50대+","Jazz":"50대+","Blues":"50대+",
        "Soul":"40대","Country":"40대","Rock":"40대",
        "Pop":"30대","R&B":"30대","Indie":"30대",
        "Hip-Hop":"20대","Rap":"20대","Electronic":"20대",
        "Dance":"10-20대","Reggaeton":"10-20대","K-Pop":"10-20대",
    }
    df['age_gen_proxy'] = df['track_genre'].map(age_map).fillna("기타")

    # 연도 → 10년 시대 구분
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['decade'] = pd.cut(df['year'],
        bins=[1999,2004,2009,2014,2019,2021],
        labels=["2000-04","2005-09","2010-14","2015-19","2020+"])

    print(f"[1] 로드 완료: {len(df):,}행  |  연도 있는 행: {df['year'].notna().sum():,}행")
    return df

# ── 2. EDA ──────────────────────────────────────────────────
def run_eda(df):
    print("\n[2] EDA")

    # Fig 01: 장르별 평균 인기도
    gp = df.groupby('track_genre')['popularity'].mean().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(13, 5))
    cols_ = [NAVY if i < 5 else "#B0BEC5" for i in range(len(gp))]
    bars = ax.bar(gp.index, gp.values, color=cols_, edgecolor='white', linewidth=.5)
    ax.set_title("장르별 평균 인기도 Top 20", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("장르"); ax.set_ylabel("평균 인기도")
    ax.tick_params(axis='x', rotation=40)
    for b in bars[:5]:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+.4,
                f"{b.get_height():.1f}", ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/01_genre_popularity.png", dpi=150); plt.close()
    print("    → 01_genre_popularity.png")

    # Fig 02: 오디오 피처 히트맵
    focus = ["Pop","Rock","Hip-Hop","Jazz","Classical","Blues","R&B","Electronic","K-Pop"]
    feats = ["danceability","energy","valence","acousticness"]
    feat_kr = {"danceability":"댄서빌리티","energy":"에너지",
               "valence":"긍정성","acousticness":"어쿠스틱성"}
    hmap = df[df['track_genre'].isin(focus)].groupby('track_genre')[feats].mean()
    hmap.columns = [feat_kr[c] for c in hmap.columns]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(hmap, annot=True, fmt='.2f', cmap='Blues',
                linewidths=.5, ax=ax, cbar_kws={'label':'평균값 (0–1)'})
    ax.set_title("장르별 오디오 피처 히트맵", fontsize=13, fontweight='bold')
    ax.set_xlabel(""); ax.set_ylabel("장르")
    plt.tight_layout()
    plt.savefig("results/02_genre_feature_heatmap.png", dpi=150); plt.close()
    print("    → 02_genre_feature_heatmap.png")

    # Fig 03: 세대 프록시별 피처
    proxy_order = ["50대+","40대","30대","20대","10-20대"]
    df_p = df[df['age_gen_proxy'].isin(proxy_order)]
    feat_info = {
        "danceability":("댄서빌리티","#4A6FA5"),
        "energy":("에너지","#E07B54"),
        "acousticness":("어쿠스틱성","#6AAB9C"),
        "valence":("긍정성","#C75E7E"),
    }
    proxy_avg = df_p.groupby('age_gen_proxy')[list(feat_info.keys())].mean().reindex(proxy_order)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (feat, (title, color)) in zip(axes.flat, feat_info.items()):
        bars = ax.bar(proxy_order, proxy_avg[feat], color=color, alpha=.85, edgecolor='white')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1); ax.set_ylabel("평균값 (0–1)")
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+.01,
                    f"{b.get_height():.2f}", ha='center', fontsize=9)
    fig.suptitle("세대별 선호 장르의 오디오 피처 비교\n(장르를 세대 프록시로 활용)",
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("results/03_age_proxy_features.png", dpi=150, bbox_inches='tight'); plt.close()
    print("    → 03_age_proxy_features.png")

# ── 3. 연도별 트렌드 (Top Hits 2000-2019) ───────────────────
def run_trend(df):
    print("\n[3] 연도별 트렌드 분석")
    df_y = df[df['year'].notna() & (df['year'] >= 2000) & (df['year'] <= 2019)].copy()
    df_y['year'] = df_y['year'].astype(int)

    # Fig 04: 연도별 댄서빌리티·에너지·어쿠스틱성 트렌드
    feats = ['danceability','energy','acousticness','valence']
    feat_kr = {'danceability':'댄서빌리티','energy':'에너지',
               'acousticness':'어쿠스틱성','valence':'긍정성'}
    yearly = df_y.groupby('year')[feats].mean()

    fig, ax = plt.subplots(figsize=(13, 5))
    colors_ = ["#4A6FA5","#E07B54","#6AAB9C","#C75E7E"]
    for feat, color in zip(feats, colors_):
        ax.plot(yearly.index, yearly[feat], marker='o', markersize=5,
                label=feat_kr[feat], color=color, linewidth=2)
    ax.set_title("2000–2019 히트곡 오디오 피처 연도별 트렌드\n(Top Hits Spotify)", fontsize=13, fontweight='bold')
    ax.set_xlabel("연도"); ax.set_ylabel("평균값 (0–1)")
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xticks(range(2000, 2020, 2))
    ax.axvspan(2000, 2009, alpha=.04, color='gray', label='2000년대')
    ax.axvspan(2010, 2019, alpha=.04, color='blue', label='2010년대')
    ax.text(2004, .08, '2000년대', ha='center', fontsize=10, color='gray', fontstyle='italic')
    ax.text(2014, .08, '2010년대', ha='center', fontsize=10, color='#4A6FA5', fontstyle='italic')
    plt.tight_layout()
    plt.savefig("results/04_yearly_trend.png", dpi=150); plt.close()
    print("    → 04_yearly_trend.png")

    # Fig 05: 연도별 장르 분포 (히트곡)
    # 다중 장르 → 첫 번째 장르만 사용
    top_genres = ['Pop','Rock','Hip-Hop','R&B','Electronic','Country','Dance','Indie']
    df_y2 = df_y[df_y['track_genre'].isin(top_genres)]
    pivot = df_y2.groupby(['year','track_genre']).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=[g for g in top_genres if g in pivot.columns])

    fig, ax = plt.subplots(figsize=(13, 5))
    pivot.plot(kind='bar', stacked=True, ax=ax,
               color=["#4A6FA5","#E07B54","#6AAB9C","#C75E7E","#8B7BB5","#D4B483","#5D9B84","#E8A838"],
               edgecolor='white', linewidth=.3)
    ax.set_title("2000–2019 히트곡 연도별 장르 분포", fontsize=13, fontweight='bold')
    ax.set_xlabel("연도"); ax.set_ylabel("곡 수")
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig("results/05_yearly_genre_dist.png", dpi=150); plt.close()
    print("    → 05_yearly_genre_dist.png")

    # Fig 06: 10년 단위 오디오 피처 비교 (2000년대 vs 2010년대)
    df_y['era'] = df_y['year'].apply(lambda x: '2000년대' if x < 2010 else '2010년대')
    era_avg = df_y.groupby('era')[feats].mean()

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, feat in zip(axes, feats):
        vals = [era_avg.loc['2000년대', feat], era_avg.loc['2010년대', feat]]
        bars = ax.bar(['2000년대','2010년대'], vals,
                      color=["#4A6FA5","#E07B54"], edgecolor='white', width=.5)
        ax.set_title(feat_kr[feat], fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1); ax.set_ylabel("평균값")
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+.01,
                    f"{b.get_height():.3f}", ha='center', fontsize=11, fontweight='bold')
        # 변화율 표시
        diff = vals[1] - vals[0]
        sign = "▲" if diff > 0 else "▼"
        color_ = "#E07B54" if diff > 0 else "#4A6FA5"
        ax.text(.5, .92, f"{sign} {abs(diff):.3f}", transform=ax.transAxes,
                ha='center', fontsize=11, color=color_, fontweight='bold')
    fig.suptitle("2000년대 vs 2010년대 히트곡 음악 특성 변화", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/06_decade_comparison.png", dpi=150); plt.close()
    print("    → 06_decade_comparison.png")

    # 통계 출력
    g00 = df_y[df_y['era']=='2000년대']
    g10 = df_y[df_y['era']=='2010년대']
    print(f"\n    2000년대 {len(g00)}곡 / 2010년대 {len(g10)}곡")
    for feat in feats:
        t, p = stats.ttest_ind(g00[feat].dropna(), g10[feat].dropna())
        print(f"    {feat_kr[feat]}: 2000s={g00[feat].mean():.3f} → 2010s={g10[feat].mean():.3f}  (t={t:.2f}, p={p:.4f})")

# ── 4. 가설 검정 ────────────────────────────────────────────
def run_hypothesis(df):
    print("\n[4] 가설 검정")
    grps_d = {
        "50대+": df[df['track_genre'].isin(["Jazz","Classical","Blues"])]["danceability"],
        "40대":  df[df['track_genre'].isin(["Rock","Soul","Country"])]["danceability"],
        "30대":  df[df['track_genre'].isin(["Pop","R&B","Indie"])]["danceability"],
        "20대":  df[df['track_genre'].isin(["Hip-Hop","Rap","Electronic"])]["danceability"],
    }
    grps_a = {k: df[df['track_genre'].isin(
        {"50대+":["Jazz","Classical","Blues"],"40대":["Rock","Soul","Country"],
         "30대":["Pop","R&B","Indie"],"20대":["Hip-Hop","Rap","Electronic"]}[k]
    )]["acousticness"] for k in grps_d}

    f1,p1 = stats.f_oneway(*grps_d.values())
    f2,p2 = stats.f_oneway(*grps_a.values())
    r, pr = stats.pearsonr(df['danceability'], df['popularity'])
    print(f"    H1 댄서빌리티 ANOVA: F={f1:.1f}, p={p1:.2e}")
    print(f"    H2 어쿠스틱성 ANOVA: F={f2:.1f}, p={p2:.2e}")
    print(f"    H3 댄서빌리티↔인기도: r={r:.3f}, p={pr:.2e}")

    # Fig 07: 세대별 박스플롯
    proxy_order = ["50대+","40대","30대","20대","10-20대"]
    df_p = df[df['age_gen_proxy'].isin(proxy_order)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=df_p, x='age_gen_proxy', y='danceability',
                order=proxy_order, palette=PALETTE, ax=axes[0])
    axes[0].set_title("세대별 댄서빌리티 분포", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("추정 연령대"); axes[0].set_ylabel("댄서빌리티 (0–1)")
    axes[0].text(.97,.97, f"ANOVA p < 0.001", transform=axes[0].transAxes,
                 ha='right', va='top', fontsize=10, color="#E07B54",
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', alpha=.8))
    sns.boxplot(data=df_p, x='age_gen_proxy', y='acousticness',
                order=proxy_order, palette=PALETTE, ax=axes[1])
    axes[1].set_title("세대별 어쿠스틱성 분포", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("추정 연령대"); axes[1].set_ylabel("어쿠스틱성 (0–1)")
    plt.tight_layout()
    plt.savefig("results/07_hypothesis_boxplots.png", dpi=150); plt.close()
    print("    → 07_hypothesis_boxplots.png")
    return f1,p1,f2,p2,r

# ── 5. 분류 모델 ────────────────────────────────────────────
def run_model(df):
    print("\n[5] 분류 모델")
    target = ["Pop","Rock","Hip-Hop","Jazz","Classical","R&B"]
    df_m = df[df['track_genre'].isin(target)].copy()
    feats = ["danceability","energy","valence","acousticness",
             "speechiness","instrumentalness","liveness","tempo"]
    feat_kr = {"danceability":"댄서빌리티","energy":"에너지","valence":"긍정성",
               "acousticness":"어쿠스틱성","speechiness":"말투","instrumentalness":"기악성",
               "liveness":"라이브감","tempo":"템포"}

    min_n = df_m['track_genre'].value_counts().min()
    df_b  = df_m.groupby('track_genre').sample(n=min(min_n, 800), random_state=42)
    le    = LabelEncoder()
    X     = df_b[feats]
    y     = le.fit_transform(df_b['track_genre'])

    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    cv  = cross_val_score(rf, X, y, cv=5).mean()
    y_p = rf.predict(X_te)
    print(f"    교차검증 정확도: {cv:.3f}")

    # Fig 08: 특성 중요도
    fi = pd.Series(rf.feature_importances_, index=feats).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    cols_ = [NAVY if v==fi.max() else "#4A6FA5" if v>fi.mean() else "#B0BEC5" for v in fi]
    bars  = ax.barh([feat_kr[f] for f in fi.index], fi.values, color=cols_)
    ax.set_title("Random Forest 특성 중요도\n(오디오 피처 → 장르 분류)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Feature Importance")
    for b in bars:
        ax.text(b.get_width()+.002, b.get_y()+b.get_height()/2,
                f"{b.get_width():.3f}", va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig("results/08_feature_importance.png", dpi=150); plt.close()
    print("    → 08_feature_importance.png")

    # Fig 09: 혼동 행렬
    cm = confusion_matrix(y_te, y_p)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(f"혼동 행렬 | 교차검증 정확도: {cv:.1%}", fontsize=12, fontweight='bold')
    ax.set_xlabel("예측 장르"); ax.set_ylabel("실제 장르")
    plt.tight_layout()
    plt.savefig("results/09_confusion_matrix.png", dpi=150); plt.close()
    print("    → 09_confusion_matrix.png")
    return cv, fi

# ── 6. 요약 산점도 ──────────────────────────────────────────
def run_summary(df):
    print("\n[6] 요약 시각화")
    focus = ["Pop","Rock","Hip-Hop","Jazz","Classical","Blues","R&B","Electronic","K-Pop"]
    avg   = df[df['track_genre'].isin(focus)].groupby('track_genre')[
        ["danceability","acousticness","popularity"]].mean()
    fig, ax = plt.subplots(figsize=(9, 6))
    colors_ = ["#4A6FA5","#E07B54","#6AAB9C","#C75E7E","#8B7BB5",
               "#D4B483","#5D9B84","#E8A838","#FF6B9D"]
    for i,(g,row) in enumerate(avg.iterrows()):
        ax.scatter(row['danceability'], row['acousticness'],
                   s=row['popularity']*8, color=colors_[i%len(colors_)],
                   alpha=.8, edgecolors='white', linewidth=1.5, zorder=5)
        ax.annotate(g,(row['danceability'],row['acousticness']),
                    textcoords='offset points', xytext=(8,4),
                    fontsize=10, fontweight='bold')
    ax.set_xlabel("댄서빌리티 — 높을수록 젊은 세대 선호", fontsize=11)
    ax.set_ylabel("어쿠스틱성 — 높을수록 고연령 선호",   fontsize=11)
    ax.set_title("장르별 음악 특성 분포\n(버블 크기 = 인기도)",
                 fontsize=13, fontweight='bold')
    ax.text(.02,.98,"↑ 고연령 선호", transform=ax.transAxes,
            va='top', fontsize=10, color="#6AAB9C", fontstyle='italic')
    ax.text(.98,.02,"젊은 세대 선호 →", transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10, color="#4A6FA5", fontstyle='italic')
    plt.tight_layout()
    plt.savefig("results/10_genre_scatter.png", dpi=150); plt.close()
    print("    → 10_genre_scatter.png")

# ── MAIN ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("연령에 따른 음악 소비 패턴 분석")
    print("김정연 | 데이터마이닝 2026-1학기")
    print("="*55)
    df = load()
    run_eda(df)
    run_trend(df)
    run_hypothesis(df)
    acc, fi = run_model(df)
    run_summary(df)
    print("\n" + "="*55)
    print(f"완료! results/ 폴더 확인")
    print(f"  분류 정확도: {acc:.1%}  |  핵심 피처: {fi.idxmax()}")
    print("="*55)
