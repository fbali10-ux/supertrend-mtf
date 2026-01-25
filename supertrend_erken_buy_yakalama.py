import os
import time
import hashlib
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from curl_cffi import requests as cffi_requests


# =======================
# 1) SİMGE LİSTESİ (SENİN LİSTE)
# =======================
SYMBOLS = [
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
"ZOREN.IS",]

# Eğer listeyi dosyada aynen tuttun diye varsayıyorum:
# Tekrarları temizle
SYMBOLS = list(dict.fromkeys([s.strip().strip('"').strip(",") for s in SYMBOLS if s and str(s).strip()]))


# =======================
# 2) PARAMETRELER
# =======================
ATR_PERIOD = 10
MULTIPLIER = 3.0

DN_DIST_LIMIT = 0.02        # %2
MIN_DN_STREAK_DAYS = 2

LOOKBACK_1H = 48            # saat
LOOKBACK_4H = 96            # saat

TOP_SCORE_MIN = 3
TOP_N = 10
VOLUME_TOP_N = 10

FAST_MODE = True            # True: daha hızlı (sleep yok)

STATE_DIR = "state"
STATE_FILE = os.path.join(STATE_DIR, "signal_hash.txt")


# =======================
# 3) ZAMAN / TELEGRAM
# =======================
def now_tr_time_str() -> str:
    tr_time = datetime.now(timezone.utc) + timedelta(hours=3)
    return tr_time.strftime("%d.%m.%Y %H:%M")


def send_telegram_message(text: str) -> None:
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram ENV yok (TELEGRAM_TOKEN / TELEGRAM_CHAT_ID). Mesaj atlanıyor.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        r = requests.post(url, json=payload, timeout=20)
        if r.status_code != 200:
            print("Telegram hata:", r.status_code, r.text[:300])
    except Exception as e:
        print("Telegram exception:", e)


# =======================
# 4) STATE HASH
# =======================
def read_prev_hash() -> str:
    if not os.path.exists(STATE_FILE):
        return ""
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def write_new_hash(new_hash: str) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        f.write(new_hash)


def stable_hash_from_dfs(df_en: pd.DataFrame, df_vol_filtered: pd.DataFrame) -> str:
    """
    Hash = EN_IYI + (ERKEN UYARI + HACİM EVET, EN_IYI hariç)
    İkisinden herhangi biri değişirse -> hash değişir -> Telegram gider
    """
    def df_to_str(df: pd.DataFrame, tag: str) -> str:
        if df is None or df.empty:
            return f"{tag}:EMPTY"
        cols = ["Hisse","MTF Skor","DN Mesafe %","Buy_1H","Buy_4H","DN Yakınlık Gün"]
        sub = df[cols].copy()
        lines = ["|".join(map(str, r)) for r in sub.itertuples(index=False)]
        return f"{tag}:" + "||".join(lines)

    s = df_to_str(df_en, "EN_IYI") + "\n" + df_to_str(df_vol_filtered, "HACIM_EVET_FILTERED")
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# =======================
# 5) VERİ ÇEKME
# =======================
def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    cols = ["Open","High","Low","Close"]
    if "Volume" in df.columns:
        cols.append("Volume")

    need = {"Open","High","Low","Close"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()

    df = df[cols].copy()
    df.dropna(subset=["Open","High","Low","Close"], inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


def safe_history(symbol: str, period: str, interval: str, max_tries: int = 4) -> pd.DataFrame:
    """
    interval SADECE: '1d', '1h', '4h'  (2h YOK!)
    """
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
        time.sleep(0.6 + i * 0.4)
    raise RuntimeError(f"{symbol} veri alınamadı: {last_err}")


# =======================
# 6) SUPER TREND (Pine v4 mantığı)
# =======================
def supertrend_pine(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    hl2 = (h + l) / 2.0

    pc = np.roll(c, 1)
    pc[0] = c[0]

    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    atr = pd.Series(tr, index=df.index).ewm(
        alpha=1 / ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD
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

        up_prev, dn_prev = up[i - 1], dn[i - 1]

        up[i] = max(up_raw, up_prev) if c[i - 1] > up_prev else up_raw
        dn[i] = min(dn_raw, dn_prev) if c[i - 1] < dn_prev else dn_raw

        if trend[i - 1] == -1 and c[i] > dn_prev:
            trend[i] = 1
        elif trend[i - 1] == 1 and c[i] < up_prev:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

    df["Trend"] = trend
    df["DN"] = dn
    df["BUY"] = (df["Trend"] == 1) & (df["Trend"].shift(1) == -1)
    return df


# =======================
# 7) ERKEN UYARI + HACİM + MTF
# =======================
def dn_distance_pct(row: pd.Series) -> float:
    if pd.isna(row.get("DN")) or pd.isna(row.get("Close")):
        return np.nan
    return (float(row["DN"]) - float(row["Close"])) / float(row["Close"])


def dn_near_streak(out: pd.DataFrame) -> int:
    cnt = 0
    for i in range(len(out) - 1, -1, -1):
        r = out.iloc[i]
        if int(r["Trend"]) != -1:
            break
        d = dn_distance_pct(r)
        if np.isnan(d) or d < 0 or d > DN_DIST_LIMIT:
            break
        cnt += 1
    return cnt


def early_warning_daily(out: pd.DataFrame):
    if out is None or len(out) < 4:
        return False, np.nan, 0

    last = out.iloc[-1]
    if int(last["Trend"]) != -1 or pd.isna(last["DN"]):
        return False, np.nan, 0

    dist = dn_distance_pct(last)
    if np.isnan(dist) or not (0 <= dist <= DN_DIST_LIMIT):
        return False, dist, 0

    c0, c1, c2 = out["Close"].iloc[-1], out["Close"].iloc[-2], out["Close"].iloc[-3]
    if not (c0 > c1 > c2):
        return False, dist, 0

    streak = dn_near_streak(out)
    if streak < MIN_DN_STREAK_DAYS:
        return False, dist, streak

    return True, dist, streak


def volume_increase_flag(df_daily: pd.DataFrame) -> bool:
    if df_daily is None or df_daily.empty or "Volume" not in df_daily.columns:
        return False
    vol = df_daily["Volume"].dropna()
    if len(vol) < 26:
        return False
    last5 = vol.iloc[-5:].mean()
    prev20 = vol.iloc[-25:-5].mean()
    if np.isnan(prev20) or prev20 <= 0:
        return False
    return last5 > prev20


def recent_buy_on_tf(symbol: str, interval: str, lookback_hours: int) -> bool:
    """
    interval SADECE: 1h ve 4h (2h kesinlikle yok)
    """
    try:
        df = safe_history(symbol, period="10d", interval=interval)
        if df.empty or len(df) < ATR_PERIOD + 5:
            return False

        out = supertrend_pine(df)
        buys = out[out["BUY"]]
        if buys.empty:
            return False

        last_buy = pd.to_datetime(buys.index[-1])
        now_utc = pd.Timestamp.now("UTC")
        if last_buy.tzinfo is None:
            last_buy = last_buy.tz_localize("UTC")

        hours_ago = (now_utc - last_buy).total_seconds() / 3600.0
        return hours_ago <= lookback_hours
    except Exception:
        return False


def sort_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.sort_values(by=["MTF Skor","DN Mesafe %"], ascending=[False, True], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_lists():
    rows_en = []
    rows_vol_yes = []

    for symbol in SYMBOLS:
        try:
            # 1) önce günlükten ele
            df_d = safe_history(symbol, period="8y", interval="1d")
            if df_d.empty or len(df_d) < (ATR_PERIOD + 10):
                continue

            out_d = supertrend_pine(df_d)
            ok, dist, streak = early_warning_daily(out_d)
            if not ok:
                continue

            # 2) hacim
            vol_yes = volume_increase_flag(df_d)

            # 3) 1h/4h teyit (sadece geçenlere)
            score = 2
            buy_1h = recent_buy_on_tf(symbol, "1h", LOOKBACK_1H)
            buy_4h = recent_buy_on_tf(symbol, "4h", LOOKBACK_4H)
            if buy_1h:
                score += 1
            if buy_4h:
                score += 1

            row = {
                "Hisse": symbol.replace(".IS",""),
                "MTF Skor": int(score),
                "DN Mesafe %": round(float(dist) * 100, 2),
                "Buy_1H": "Evet" if buy_1h else "Hayır",
                "Buy_4H": "Evet" if buy_4h else "Hayır",
                "DN Yakınlık Gün": int(streak),
            }

            if vol_yes:
                rows_vol_yes.append(row)

            if score >= TOP_SCORE_MIN:
                rows_en.append(row)

            if not FAST_MODE:
                time.sleep(0.15)

        except Exception:
            continue

    df_en = sort_df(pd.DataFrame(rows_en)).head(TOP_N)
    df_vol = sort_df(pd.DataFrame(rows_vol_yes)).head(VOLUME_TOP_N)
    return df_en, df_vol


def build_telegram_message(df_en: pd.DataFrame, df_vol_filtered: pd.DataFrame) -> str:
    ts = now_tr_time_str()
    msg = f"🕒 <i>{ts}</i>\n\n"

    msg += f"📈 <b>EN_IYI (Skor ≥ {TOP_SCORE_MIN})</b>\n"
    if df_en.empty:
        msg += "⏳ Yok\n\n"
    else:
        for _, r in df_en.iterrows():
            msg += (
                f"• <b>{r['Hisse']}</b> | Skor {r['MTF Skor']} | DN% {r['DN Mesafe %']} | "
                f"1H:{r['Buy_1H']} | 4H:{r['Buy_4H']} | Streak:{r['DN Yakınlık Gün']}\n"
            )
        msg += f"\nToplam: <b>{len(df_en)}</b>\n\n"

    msg += "🔥 <b>ERKEN UYARI + HACİM EVET (EN_IYI hariç)</b>\n"
    if df_vol_filtered.empty:
        msg += "⏳ Yok\n"
    else:
        for _, r in df_vol_filtered.iterrows():
            msg += (
                f"• <b>{r['Hisse']}</b> | Skor {r['MTF Skor']} | DN% {r['DN Mesafe %']} | "
                f"1H:{r['Buy_1H']} | 4H:{r['Buy_4H']} | Streak:{r['DN Yakınlık Gün']}\n"
            )
        msg += f"\nToplam: <b>{len(df_vol_filtered)}</b>\n"

    return msg


def main():
    df_en, df_vol = build_lists()

    # HACİM listesinden EN_IYI tekrarlarını çıkar
    en_set = set(df_en["Hisse"].tolist()) if (df_en is not None and not df_en.empty) else set()
    df_vol_filtered = df_vol[~df_vol["Hisse"].isin(en_set)].copy() if (df_vol is not None and not df_vol.empty) else pd.DataFrame()

    prev_hash = read_prev_hash()
    new_hash = stable_hash_from_dfs(df_en, df_vol_filtered)

    print("Prev hash:", prev_hash)
    print("New  hash:", new_hash)

    # İkisinden biri değişirse gönder
    if new_hash != prev_hash:
        msg = build_telegram_message(df_en, df_vol_filtered)
        send_telegram_message(msg)
        write_new_hash(new_hash)
        print("Değişim var → Telegram gönderildi, state güncellendi.")
    else:
        print("Değişim yok → Telegram gönderilmedi.")


if __name__ == "__main__":
    main()
