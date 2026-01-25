import time
import random
import pandas as pd
import numpy as np
import yfinance as yf
from curl_cffi import requests as cffi_requests


# =======================
# AYARLAR
# =======================
BIST30 = [
"A1YEN.IS",
"ACSEL.IS",
"ADEL.IS",
"ADESE.IS",
"ADGYO.IS",
"AEFES.IS",
"AFYON.IS",
"AGESA.IS",
"AGHOL.IS",
"AGROT.IS",
"AGYO.IS",
"AHGAZ.IS",
"AHSGY.IS",
"AKBNK.IS",
"AKCNS.IS",
"AKENR.IS",
"AKFGY.IS",
"AKFIS.IS",
"AKFYE.IS",
"AKGRT.IS",
"AKMGY.IS",
"AKSA.IS",
"AKSEN.IS",
"AKSGY.IS",
"AKSUE.IS",
"AKYHO.IS",
"ALARK.IS",
"ALBRK.IS",
"ALCAR.IS",
"ALCTL.IS",
"ALFAS.IS",
"ALGYO.IS",
"ALKA.IS",
"ALKIM.IS",
"ALKLC.IS",
"ALTNY.IS",
"ALVES.IS",
"ANELE.IS",
"ANGEN.IS",
"ANHYT.IS",
"ANSGR.IS",
"APBDL.IS",
"APLIB.IS",
"APMDL.IS",
"ARASE.IS",
"ARCLK.IS",
"ARDYZ.IS",
"ARENA.IS",
"ARFYE.IS",
"ARMGD.IS",
"ARSAN.IS",
"ARTMS.IS",
"ARZUM.IS",
"ASELS.IS",
"ASGYO.IS",
"ASTOR.IS",
"ASUZU.IS",
"ATAGY.IS",
"ATAKP.IS",
"ATATP.IS",
"ATEKS.IS",
"ATLAS.IS",
"ATSYH.IS",
"AVGYO.IS",
"AVHOL.IS",
"AVOD.IS",
"AVPGY.IS",
"AVTUR.IS",
"AYCES.IS",
"AYDEM.IS",
"AYEN.IS",
"AYES.IS",
"AYGAZ.IS",
"AZTEK.IS",
"BAGFS.IS",
"BAHKM.IS",
"BAKAB.IS",
"BALAT.IS",
"BALSU.IS",
"BANVT.IS",
"BARMA.IS",
"BASCM.IS",
"BASGZ.IS",
"BAYRK.IS",
"BEGYO.IS",
"BERA.IS",
"BESLR.IS",
"BEYAZ.IS",
"BFREN.IS",
"BIENY.IS",
"BIGCH.IS",
"BIGEN.IS",
"BIGTK.IS",
"BIMAS.IS",
"BINBN.IS",
"BINHO.IS",
"BIOEN.IS",
"BIZIM.IS",
"BJKAS.IS",
"BLCYT.IS",
"BLUME.IS",
"BMSCH.IS",
"BMSTL.IS",
"BNTAS.IS",
"BOBET.IS",
"BORLS.IS",
"BORSK.IS",
"BOSSA.IS",
"BRISA.IS",
"BRKO.IS",
"BRKSN.IS",
"BRKVY.IS",
"BRLSM.IS",
"BRMEN.IS",
"BRSAN.IS",
"BRYAT.IS",
"BSOKE.IS",
"BTCIM.IS",
"BUCIM.IS",
"BULGS.IS",
"BURCE.IS",
"BURVA.IS",
"BVSAN.IS",
"BYDNR.IS",
"CANTE.IS",
"CASA.IS",
"CATES.IS",
"CCOLA.IS",
"CELHA.IS",
"CEMAS.IS",
"CEMTS.IS",
"CEMZY.IS",
"CEOEM.IS",
"CGCAM.IS",
"CIMSA.IS",
"CLEBI.IS",
"CMBTN.IS",
"CMENT.IS",
"CONSE.IS",
"COSMO.IS",
"CRDFA.IS",
"CRFSA.IS",
"CUSAN.IS",
"CVKMD.IS",
"CWENE.IS",
"DAGI.IS",
"DAPGM.IS",
"DARDL.IS",
"DCTTR.IS",
"DENGE.IS",
"DERHL.IS",
"DERIM.IS",
"DESA.IS",
"DESPC.IS",
"DEVA.IS",
"DGATE.IS",
"DGGYO.IS",
"DGNMO.IS",
"DIRIT.IS",
"DITAS.IS",
"DMRGD.IS",
"DMSAS.IS",
"DNISI.IS",
"DOAS.IS",
"DOCO.IS",
"DOFER.IS",
"DOFRB.IS",
"DOGUB.IS",
"DOHOL.IS",
"DOKTA.IS",
"DSTKF.IS",
"DUNYH.IS",
"DURDO.IS",
"DURKN.IS",
"DYOBY.IS",
"DZGYO.IS",
"EBEBK.IS",
"ECILC.IS",
"ECOGR.IS",
"ECZYT.IS",
"EDATA.IS",
"EDIP.IS",
"EFOR.IS",
"EGEEN.IS",
"EGEGY.IS",
"EGEPO.IS",
"EGGUB.IS",
"EGPRO.IS",
"EGSER.IS",
"EKGYO.IS",
"EKIZ.IS",
"EKOS.IS",
"EKSUN.IS",
"ELITE.IS",
"EMKEL.IS",
"EMNIS.IS",
"ENDAE.IS",
"ENERY.IS",
"ENJSA.IS",
"ENKAI.IS",
"ENSRI.IS",
"ENTRA.IS",
"EPLAS.IS",
"ERBOS.IS",
"ERCB.IS",
"EREGL.IS",
"ERSU.IS",
"ESCAR.IS",
"ESCOM.IS",
"ESEN.IS",
"ETILR.IS",
"ETYAT.IS",
"EUHOL.IS",
"EUKYO.IS",
"EUPWR.IS",
"EUREN.IS",
"EUYO.IS",
"EYGYO.IS",
"FADE.IS",
"FENER.IS",
"FLAP.IS",
"FMIZP.IS",
"FONET.IS",
"FORMT.IS",
"FORTE.IS",
"FRIGO.IS",
"FROTO.IS",
"FZLGY.IS",
"GARAN.IS",
"GARFA.IS",
"GEDIK.IS",
"GEDZA.IS",
"GENIL.IS",
"GENTS.IS",
"GEREL.IS",
"GESAN.IS",
"GIPTA.IS",
"GLBMD.IS",
"GLCVY.IS",
"GLDTR.IS",
"GLRMK.IS",
"GLRYH.IS",
"GLYHO.IS",
"GMSTR.IS",
"GMTAS.IS",
"GOKNR.IS",
"GOLTS.IS",
"GOODY.IS",
"GOZDE.IS",
"GRNYO.IS",
"GRSEL.IS",
"GRTHO.IS",
"GSDDE.IS",
"GSDHO.IS",
"GSRAY.IS",
"GUBRF.IS",
"GUNDG.IS",
"GWIND.IS",
"GZNMI.IS",
"HALKB.IS",
"HALKS.IS",
"HATEK.IS",
"HATSN.IS",
"HDFGS.IS",
"HEDEF.IS",
"HEKTS.IS",
"HKTM.IS",
"HLGYO.IS",
"HOROZ.IS",
"HRKET.IS",
"HTTBT.IS",
"HUBVC.IS",
"HUNER.IS",
"HURGZ.IS",
"ICBCT.IS",
"ICUGS.IS",
"IDGYO.IS",
"IEYHO.IS",
"IHAAS.IS",
"IHEVA.IS",
"IHGZT.IS",
"IHLAS.IS",
"IHLGM.IS",
"IHYAY.IS",
"IMASM.IS",
"INDES.IS",
"INFO.IS",
"INGRM.IS",
"INTEK.IS",
"INTEM.IS",
"INVEO.IS",
"INVES.IS",
"ISATR.IS",
"ISBIR.IS",
"ISBTR.IS",
"ISCTR.IS",
"ISDMR.IS",
"ISFIN.IS",
"ISGLK.IS",
"ISGSY.IS",
"ISGYO.IS",
"ISIST.IS",
"ISKPL.IS",
"ISKUR.IS",
"ISMEN.IS",
"ISSEN.IS",
"ISYAT.IS",
"IZENR.IS",
"IZFAS.IS",
"IZINV.IS",
"IZMDC.IS",
"JANTS.IS",
"KAPLM.IS",
"KAREL.IS",
"KARSN.IS",
"KARTN.IS",
"KATMR.IS",
"KAYSE.IS",
"KBORU.IS",
"KCAER.IS",
"KCHOL.IS",
"KENT.IS",
"KERVN.IS",
"KFEIN.IS",
"KGYO.IS",
"KIMMR.IS",
"KLGYO.IS",
"KLKIM.IS",
"KLMSN.IS",
"KLNMA.IS",
"KLRHO.IS",
"KLSER.IS",
"KLSYN.IS",
"KLYPV.IS",
"KMPUR.IS",
"KNFRT.IS",
"KOCMT.IS",
"KONKA.IS",
"KONTR.IS",
"KONYA.IS",
"KOPOL.IS",
"KORDS.IS",
"KOTON.IS",
"KRDMA.IS",
"KRDMB.IS",
"KRDMD.IS",
"KRGYO.IS",
"KRONT.IS",
"KRPLS.IS",
"KRSTL.IS",
"KRTEK.IS",
"KRVGD.IS",
"KSTUR.IS",
"KTLEV.IS",
"KTSKR.IS",
"KUTPO.IS",
"KUVVA.IS",
"KUYAS.IS",
"KZBGY.IS",
"KZGYO.IS",
"LIDER.IS",
"LIDFA.IS",
"LILAK.IS",
"LINK.IS",
"LKMNH.IS",
"LMKDC.IS",
"LOGO.IS",
"LRSHO.IS",
"LUKSK.IS",
"LYDHO.IS",
"LYDYE.IS",
"MAALT.IS",
"MACKO.IS",
"MAGEN.IS",
"MAKIM.IS",
"MAKTK.IS",
"MANAS.IS",
"MARBL.IS",
"MARKA.IS",
"MARMR.IS",
"MARTI.IS",
"MAVI.IS",
"MEDTR.IS",
"MEGAP.IS",
"MEGMT.IS",
"MEKAG.IS",
"MEPET.IS",
"MERCN.IS",
"MERIT.IS",
"MERKO.IS",
"METRO.IS",
"MGROS.IS",
"MHRGY.IS",
"MIATK.IS",
"MMCAS.IS",
"MNDRS.IS",
"MNDTR.IS",
"MOBTL.IS",
"MOGAN.IS",
"MOPAS.IS",
"MPARK.IS",
"MRGYO.IS",
"MRSHL.IS",
"MSGYO.IS",
"MTRKS.IS",
"MTRYO.IS",
"MZHLD.IS",
"NATEN.IS",
"NETAS.IS",
"NIBAS.IS",
"NTGAZ.IS",
"NTHOL.IS",
"NUGYO.IS",
"NUHCM.IS",
"OBAMS.IS",
"OBASE.IS",
"ODAS.IS",
"ODINE.IS",
"OFSYM.IS",
"ONCSM.IS",
"ONRYT.IS",
"OPK30.IS",
"OPT25.IS",
"OPTGY.IS",
"OPTLR.IS",
"ORCAY.IS",
"ORGE.IS",
"ORMA.IS",
"OSMEN.IS",
"OSTIM.IS",
"OTKAR.IS",
"OTTO.IS",
"OYAKC.IS",
"OYAYO.IS",
"OYLUM.IS",
"OYYAT.IS",
"OZATD.IS",
"OZGYO.IS",
"OZKGY.IS",
"OZRDN.IS",
"OZSUB.IS",
"OZYSR.IS",
"PAGYO.IS",
"PAHOL.IS",
"PAMEL.IS",
"PAPIL.IS",
"PARSN.IS",
"PASEU.IS",
"PATEK.IS",
"PCILT.IS",
"PEKGY.IS",
"PENGD.IS",
"PENTA.IS",
"PETKM.IS",
"PETUN.IS",
"PGSUS.IS",
"PINSU.IS",
"PKART.IS",
"PKENT.IS",
"PLTUR.IS",
"PNLSN.IS",
"PNSUT.IS",
"POLHO.IS",
"POLTK.IS",
"PRDGS.IS",
"PRKAB.IS",
"PRKME.IS",
"PRZMA.IS",
"PSDTC.IS",
"PSGYO.IS",
"QNBFK.IS",
"QNBTR.IS",
"QTEMZ.IS",
"QUAGR.IS",
"RALYH.IS",
"RAYSG.IS",
"REEDR.IS",
"RGYAS.IS",
"RNPOL.IS",
"RODRG.IS",
"ROYAL.IS",
"RTALB.IS",
"RUBNS.IS",
"RUZYE.IS",
"RYGYO.IS",
"RYSAS.IS",
"SAFKR.IS",
"SAHOL.IS",
"SAMAT.IS",
"SANEL.IS",
"SANFM.IS",
"SANKO.IS",
"SARKY.IS",
"SASA.IS",
"SAYAS.IS",
"SDTTR.IS",
"SEGMN.IS",
"SEGYO.IS",
"SEKFK.IS",
"SEKUR.IS",
"SELEC.IS",
"SELVA.IS",
"SERNT.IS",
"SEYKM.IS",
"SILVR.IS",
"SISE.IS",
"SKBNK.IS",
"SKTAS.IS",
"SKYLP.IS",
"SKYMD.IS",
"SMART.IS",
"SMRTG.IS",
"SMRVA.IS",
"SNGYO.IS",
"SNICA.IS",
"SNKRN.IS",
"SNPAM.IS",
"SODSN.IS",
"SOKE.IS",
"SOKM.IS",
"SONME.IS",
"SRVGY.IS",
"SUMAS.IS",
"SUNTK.IS",
"SURGY.IS",
"SUWEN.IS",
"TABGD.IS",
"TARKM.IS",
"TATEN.IS",
"TATGD.IS",
"TAVHL.IS",
"TBORG.IS",
"TCELL.IS",
"TCKRC.IS",
"TDGYO.IS",
"TEHOL.IS",
"TEKTU.IS",
"TERA.IS",
"TEZOL.IS",
"TGSAS.IS",
"THYAO.IS",
"TKFEN.IS",
"TKNSA.IS",
"TLMAN.IS",
"TMPOL.IS",
"TMSN.IS",
"TNZTP.IS",
"TOASO.IS",
"TRALT.IS",
"TRCAS.IS",
"TRENJ.IS",
"TRGYO.IS",
"TRHOL.IS",
"TRILC.IS",
"TRMET.IS",
"TSGYO.IS",
"TSKB.IS",
"TSPOR.IS",
"TTKOM.IS",
"TTRAK.IS",
"TUCLK.IS",
"TUKAS.IS",
"TUPRS.IS",
"TUREX.IS",
"TURGG.IS",
"TURSG.IS",
"UFUK.IS",
"ULAS.IS",
"ULKER.IS",
"ULUFA.IS",
"ULUSE.IS",
"ULUUN.IS",
"UNLU.IS",
"USAK.IS",
"USDTR.IS",
"VAKBN.IS",
"VAKFA.IS",
"VAKFN.IS",
"VAKKO.IS",
"VANGD.IS",
"VBTYZ.IS",
"VERTU.IS",
"VERUS.IS",
"VESBE.IS",
"VESTL.IS",
"VKFYO.IS",
"VKGYO.IS",
"VKING.IS",
"VRGYO.IS",
"VSNMD.IS",
"YAPRK.IS",
"YATAS.IS",
"YAYLA.IS",
"YBTAS.IS",
"YEOTK.IS",
"YESIL.IS",
"YGGYO.IS",
"YGYO.IS",
"YIGIT.IS",
"YKBNK.IS",
"YKSLN.IS",
"YONGA.IS",
"YUNSA.IS",
"YYAPI.IS",
"YYLGD.IS",
"ZEDUR.IS",
"ZELOT.IS",
"ZERGY.IS",
"ZGOLD.IS",
"ZOREN.IS",
]

ATR_PERIOD = 10
MULTIPLIER = 3.0

DN_DIST_LIMIT = 0.02        # %2
MIN_DN_STREAK_DAYS = 2

LOOKBACK_1H = 48            # saat
LOOKBACK_4H = 96            # saat

TOP_SCORE_MIN = 3
TOP_N = 10


# =======================
# YARDIMCI
# =======================
def tradingview_link(symbol: str) -> str:
    """
    EN_IYI sheet için tıklanabilir TradingView linki
    (BIST günlük grafik)
    """
    sym = symbol.replace(".IS", "")
    return f"https://www.tradingview.com/chart/?symbol=BIST:{sym}"


# =======================
# VERİ ÇEKME
# =======================
def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    cols = ["Open","High","Low","Close"]
    if "Volume" in df.columns:
        cols.append("Volume")

    missing = [c for c in ["Open","High","Low","Close"] if c not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df[cols].copy()
    df.dropna(subset=["Open","High","Low","Close"], inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


def safe_history(symbol: str, period: str, interval: str, max_tries: int = 5) -> pd.DataFrame:
    last_err = None
    for i in range(max_tries):
        try:
            sess = cffi_requests.Session(impersonate="chrome")
            t = yf.Ticker(symbol, session=sess)
            df = t.history(period=period, interval=interval, auto_adjust=False)
            df = normalize_ohlc(df)
            if not df.empty:
                return df
        except Exception as e:
            last_err = e
        time.sleep(1.2 + i + random.uniform(0.0, 0.4))
    raise RuntimeError(f"{symbol} veri alınamadı: {last_err}")


# =======================
# SUPER TREND (Pine uyumlu)
# =======================
def supertrend_pine(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    hl2 = (h + l) / 2.0

    pc = np.roll(c, 1)
    pc[0] = c[0]

    tr = np.maximum.reduce([h-l, np.abs(h-pc), np.abs(l-pc)])
    atr = pd.Series(tr, index=df.index).ewm(
        alpha=1/ATR_PERIOD,
        adjust=False,
        min_periods=ATR_PERIOD
    ).mean().values

    n = len(df)
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    trend = np.ones(n, dtype=int)

    for i in range(n):
        if np.isnan(atr[i]):
            continue

        up_raw = hl2[i] - MULTIPLIER * atr[i]
        dn_raw = hl2[i] + MULTIPLIER * atr[i]

        if i == 0:
            up[i], dn[i] = up_raw, dn_raw
            continue

        up_prev, dn_prev = up[i-1], dn[i-1]

        up[i] = max(up_raw, up_prev) if c[i-1] > up_prev else up_raw
        dn[i] = min(dn_raw, dn_prev) if c[i-1] < dn_prev else dn_raw

        if trend[i-1] == -1 and c[i] > dn_prev:
            trend[i] = 1
        elif trend[i-1] == 1 and c[i] < up_prev:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]

    df["Trend"] = trend
    df["UP"] = up
    df["DN"] = dn
    df["BUY"] = (df["Trend"] == 1) & (df["Trend"].shift(1) == -1)
    return df


# =======================
# ERKEN UYARI HESAPLARI
# =======================
def dn_distance_pct(row: pd.Series) -> float:
    if pd.isna(row.get("DN")) or pd.isna(row.get("Close")):
        return np.nan
    return (row["DN"] - row["Close"]) / row["Close"]


def dn_near_streak(out: pd.DataFrame) -> int:
    cnt = 0
    for i in range(len(out)-1, -1, -1):
        r = out.iloc[i]
        if int(r["Trend"]) != -1:
            break
        dist = dn_distance_pct(r)
        if np.isnan(dist) or dist < 0 or dist > DN_DIST_LIMIT:
            break
        cnt += 1
    return cnt


def volume_increase_flag(out: pd.DataFrame) -> bool:
    if "Volume" not in out.columns:
        return False
    vol = out["Volume"].dropna()
    if len(vol) < 26:
        return False
    return vol.iloc[-5:].mean() > vol.iloc[-25:-5].mean()


def early_warning_daily(out: pd.DataFrame):
    if len(out) < 4:
        return False, np.nan, 0

    last = out.iloc[-1]
    if int(last["Trend"]) != -1 or pd.isna(last["DN"]):
        return False, np.nan, 0

    dist = dn_distance_pct(last)
    if np.isnan(dist) or dist < 0 or dist > DN_DIST_LIMIT:
        return False, dist, 0

    c0, c1, c2 = out["Close"].iloc[-1], out["Close"].iloc[-2], out["Close"].iloc[-3]
    if not (c0 > c1 > c2):
        return False, dist, 0

    streak = dn_near_streak(out)
    if streak < MIN_DN_STREAK_DAYS:
        return False, dist, streak

    return True, dist, streak


# =======================
# MTF BUY TEYİDİ
# =======================
def recent_buy_on_tf(symbol: str, interval: str, lookback_hours: int) -> bool:
    try:
        df = safe_history(symbol, period="10d", interval=interval)
        out = supertrend_pine(df)
        buys = out[out["BUY"]]
        if buys.empty:
            return False

        last_buy = pd.to_datetime(buys.index[-1])
        now_utc = pd.Timestamp.now("UTC")
        if last_buy.tzinfo is None:
            last_buy = last_buy.tz_localize("UTC")

        hours_ago = (now_utc - last_buy).total_seconds() / 3600
        return hours_ago <= lookback_hours
    except Exception:
        return False


# =======================
# MAIN
# =======================
def main():
    erken_evet, erken_hayir, son_buy, hatalar = [], [], [], []

    for symbol in BIST30:
        print(f"Taraniyor: {symbol}")
        try:
            df_d = safe_history(symbol, "8y", "1d")
            out_d = supertrend_pine(df_d)

            # SON BUY
            buys = out_d[out_d["BUY"]]
            if not buys.empty:
                lb = buys.iloc[-1]
                son_buy.append({
                    "Hisse": symbol.replace(".IS",""),
                    "Son Buy Tarihi": lb.name.date(),
                    "Son Buy Fiyatı": round(lb["Close"],2)
                })

            ok, dist, streak = early_warning_daily(out_d)
            if not ok:
                continue

            vol_flag = volume_increase_flag(out_d)

            score = 2
            buy_1h = recent_buy_on_tf(symbol, "1h", LOOKBACK_1H)
            buy_4h = recent_buy_on_tf(symbol, "4h", LOOKBACK_4H)

            if buy_1h: score += 1
            if buy_4h: score += 1

            row = {
                "Hisse": symbol.replace(".IS",""),
                "Kapanış": round(out_d.iloc[-1]["Close"],2),
                "DN Bandı": round(out_d.iloc[-1]["DN"],2),
                "DN Mesafe %": round(dist*100,2),
                "DN Yakınlık Gün": streak,
                "1H Buy": "Evet" if buy_1h else "Hayır",
                "4H Buy": "Evet" if buy_4h else "Hayır",
                "MTF Skor": score,
                "Hacim Artışı": "Evet" if vol_flag else "Hayır"
            }

            (erken_evet if vol_flag else erken_hayir).append(row)

        except Exception as e:
            hatalar.append((symbol, str(e)))

    df_evet = pd.DataFrame(erken_evet)
    df_hayir = pd.DataFrame(erken_hayir)
    df_son = pd.DataFrame(son_buy).sort_values("Son Buy Tarihi", ascending=False)
    df_hat = pd.DataFrame(hatalar, columns=["Hisse","Hata"])

    if not df_evet.empty:
        df_evet.sort_values(["MTF Skor","DN Mesafe %"], ascending=[False,True], inplace=True)
    if not df_hayir.empty:
        df_hayir.sort_values(["MTF Skor","DN Mesafe %"], ascending=[False,True], inplace=True)

    df_all = pd.concat([df_evet, df_hayir], ignore_index=True)
    df_en_iyi = df_all[df_all["MTF Skor"] >= TOP_SCORE_MIN] \
        .sort_values(["MTF Skor","DN Mesafe %"], ascending=[False,True]) \
        .head(TOP_N)

    # TradingView linki EKLE
    if not df_en_iyi.empty:
        df_en_iyi["TradingView"] = df_en_iyi["Hisse"].apply(tradingview_link)

    with pd.ExcelWriter("bist30_supertrend_mtf_erken_uyari.xlsx", engine="openpyxl") as writer:
        df_en_iyi.to_excel(writer, index=False, sheet_name="EN_IYI")
        df_evet.to_excel(writer, index=False, sheet_name="ERKEN_UYARI_HACIM_EVET")
        df_hayir.to_excel(writer, index=False, sheet_name="ERKEN_UYARI_HACIM_HAYIR")
        df_son.to_excel(writer, index=False, sheet_name="SON_BUY")
        df_hat.to_excel(writer, index=False, sheet_name="HATALAR")

    print("\n✅ bist30_supertrend_mtf_erken_uyari.xlsx oluşturuldu")


if __name__ == "__main__":
    main()
