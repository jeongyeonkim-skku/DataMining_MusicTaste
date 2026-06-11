# 🎵 연령에 따른 음악 소비 패턴 분석

> **Data Mining Final Project** | 2026년 1학기 | 미래안보AI학과 김정연

**GitHub**: https://github.com/jeongyeonkim-skku/DataMining_MusicTaste

---

## 연구 질문
> 연령과 음악 소비 패턴 간에는 상관 관계가 있는가?  
> 오디오 피처(댄서빌리티, 어쿠스틱성 등)로 세대별 음악 취향을 설명할 수 있는가?

## 핵심 전략 (피드백 반영)
Spotify 트랙 데이터에는 연령 정보가 없으므로 **장르를 세대 프록시(Proxy)**로 활용:
| 장르 | 추정 연령대 |
|------|-----------|
| Classical, Jazz, Blues | 50대+ |
| Rock, Country, Soul | 40대 |
| Pop, R&B, Indie | 30대 |
| Hip-Hop, Rap, Electronic | 20대 |

## 실행 방법

```bash
pip install -r requirements.txt
python analysis.py
# → results/ 폴더에 시각화 8개 자동 생성
```

## 주요 결과
- **가설 검정**: 세대별 댄서빌리티(F=8,381, p<0.001), 어쿠스틱성(F=9,905, p<0.001) 유의미한 차이
- **Random Forest 분류**: 교차검증 정확도 54.0% (6개 장르 중 Classical 90% 정밀도)
- **가장 중요한 피처**: Energy > Danceability > Acousticness

## 데이터
- **Kaggle Spotify DB**: 232,725 트랙, 18개 변수 → `data/SpotifyFeatures.csv`
- (보조) Last.fm API, 설문조사 (연령 직접 변수 확보 목표)

## 한계 및 향후 개선
- 연령 직접 데이터 부재로 간접 추론 방식 사용
- Popularity 변수의 최신곡 편향 존재
- 설문 100명+ 확보 후 직접 연령-피처 회귀분석 예정
