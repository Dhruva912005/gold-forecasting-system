import sqlite3
import pandas as pd
import yfinance as yf
import feedparser
import urllib.parse
from dateutil import parser
from datetime import datetime, timedelta
import asyncio
import aiohttp
import nest_asyncio
import os
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "gold_data.db")
CSV_PATH = os.path.join(BASE_DIR, "data.csv")

nest_asyncio.apply()

# --------------------------------
# EXTENSIVE KEYWORDS (Converted to Gold)
# --------------------------------
ALL_KEYWORDS = [
    # 🪙 GOLD CORE
    "gold price","gold market","gold demand","gold supply",
    "gold investment","gold trading","gold ETF","gold futures",
    "gold safe haven","gold inflation hedge","gold spot price",
    "gold price outlook","gold market outlook","gold price forecast",
    "gold demand trends","gold supply outlook","gold ETF flows",
    "gold futures price","gold investment demand","gold price update",
    "gold market news",

    # 📈 PRICE ACTION
    "gold price rally","gold price surge","gold price drop",
    "gold price volatility","gold price trend","gold price movement",
    "gold price pressure","gold price rebound","gold price decline",

    # 📈 TECHNICAL / ADVANCED
    "price breakout","price breakdown","support level","resistance level",
    "trend reversal","bullish trend","bearish trend","sideways market",
    "price consolidation","momentum trading","technical indicators",
    "relative strength index","moving averages","MACD indicator",

    # 🌍 MACRO
    "inflation data","inflation trends","inflation outlook",
    "consumer price index","core inflation","inflation expectations",
    "inflation shock","disinflation trends","stagflation risk",
    "interest rates","fed rate decision","rate hike",
    "central bank policy","monetary policy",

    # 🏛️ CENTRAL BANK
    "fed policy","central bank update","policy rate outlook",
    "rate cut expectations","rate hike expectations",
    "policy guidance","monetary policy outlook",
    "fed interest rate path","ECB policy decision","RBI policy update",
    "central bank intervention","policy divergence",
    "rate pause expectations","forward guidance signals",

    # 🏭 INDUSTRIAL / GREEN DEMAND
    "gold industrial demand","gold electronics demand",
    "gold manufacturing demand","gold industrial use",

    # ⚡ ENERGY / COMMODITY
    "oil price","crude oil","energy market","fuel price increase",
    "energy crisis","oil price surge","commodity market",
    "commodity prices","metal prices","industrial metals demand",
    "global commodity cycle",

    # 🌐 GEOPOLITICS
    "geopolitical tensions","global conflict","war impact economy",
    "russia ukraine war","geopolitical risk sentiment",
    "war driven uncertainty","trade war impact",

    # 🌍 GLOBAL TRADE
    "trade deficit","trade surplus","export demand",
    "import trends","supply chain disruption",
    "global logistics crisis",

    # 📊 MARKET
    "stock market","equity market","market trends",
    "market volatility","market outlook","market correction",
    "market selloff","global market trends","market uncertainty",
    "market analysis","market report","market overview",

    # 📊 MARKET MICROSTRUCTURE
    "order book depth","bid ask spread","market efficiency",
    "high frequency trading","algorithmic trading",
    "liquidity crunch","flash crash","market manipulation",

    # 🏦 ECONOMY
    "economic growth","economic outlook","economic slowdown",
    "GDP growth","recession risk","economic data",
    "global economy","world economy","global growth",
    "economic indicators","economic forecast",
    "economic contraction","consumer spending slowdown",
    "business cycle downturn",

    # 💰 CAPITAL FLOWS
    "capital flows","investment flows","fund flows",
    "ETF flows","portfolio allocation","asset allocation",
    "institutional investment flows","foreign investment flows",
    "risk on flows","risk off flows","capital rotation",
    "sector rotation","smart money flows","hedge fund positioning",
    "institutional allocation","retail investor flows",

    # 📊 VOLATILITY
    "volatility index","market volatility index",
    "market swings","price volatility trends",
    "market turbulence","volatile markets",

    # 📊 MACRO DATA
    "PMI data","manufacturing PMI","services PMI",
    "employment data","labor market trends",
    "retail sales","industrial production",

    # 🌍 GLOBAL MACRO
    "global economic outlook","global recession risk",
    "global inflation trends","global monetary policy",
    "international trade","emerging markets"
]

async def fetch_feed(session, url):
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                return await response.text()
    except Exception:
        return None
    return None

async def fetch_articles_for_date(session, target_date, keywords):
    titles = []
    
    for word in keywords:
        query = urllib.parse.quote(word)
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        feed_text = await fetch_feed(session, url)
        
        if feed_text:
            try:
                feed = feedparser.parse(feed_text)
                for entry in feed.entries:
                    try:
                        publish_date = parser.parse(entry.published).date()
                        
                        if publish_date == target_date.date():
                            title = entry.title
                            if title not in titles:
                                titles.append(title)
                                
                        if len(titles) == 5:
                            return titles
                    except:
                        continue
            except:
                pass

    while len(titles) < 5:
        titles.append("No Article")
        
    return titles[:5]

async def process_all_dates(missing_dates, keywords):
    articles_dict = {}
    async with aiohttp.ClientSession() as session:
        tasks = []
        for date in missing_dates:
            tasks.append(fetch_articles_for_date(session, date, keywords))
            
        all_results = await asyncio.gather(*tasks)
        
        for i, date in enumerate(missing_dates):
            articles_dict[date] = all_results[i]
            
    return articles_dict

def update_dataset():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    
    try:
        existing = pd.read_sql("SELECT * FROM gold_prices ORDER BY Date ASC", conn)
        existing["Date"] = pd.to_datetime(existing["Date"])
        last_date = existing["Date"].max()
        # Always re-verify the last 3 business days to overwrite placeholders
        start_date = last_date - (3 * us_bd)
    except:
        existing = pd.DataFrame()
        start_date = datetime(2022, 4, 1)

    today_date = pd.Timestamp.today().normalize()
    
    required_dates = pd.date_range(start=start_date, end=today_date, freq=us_bd)

    if len(required_dates) == 0:
        print(f"Data is already perfectly synchronized to today ({today_date.date()}).")
        conn.close()
        return existing

    print(f"Force-Fetching specifically generated array from: {start_date.date()} to {today_date.date()}...")
    
    gold = yf.download("GC=F", start=start_date.strftime("%Y-%m-%d"), end=(today_date + timedelta(days=1)).strftime("%Y-%m-%d"), progress=False)
    
    if not gold.empty:
        gold = gold[['Close']]
        gold.columns = ["Gold_Price"]
        gold.index = gold.index.normalize()
    else:
        gold = pd.DataFrame(columns=["Gold_Price"])

    gold_df = pd.DataFrame(index=required_dates)
    gold_df.index.name = "Date"
    
    gold_final = gold_df.merge(gold, left_index=True, right_index=True, how='left')
    gold_final = gold_final.reset_index()

    if not existing.empty:
        gold_final.loc[0, "Gold_Price"] = gold_final.loc[0, "Gold_Price"] if not pd.isna(gold_final.loc[0, "Gold_Price"]) else existing["Gold_Price"].iloc[-1]
    
    gold_final["Gold_Price"] = gold_final["Gold_Price"].ffill()
    gold_final["Gold_Price"] = gold_final["Gold_Price"].fillna(gold_final["Gold_Price"].rolling(3, min_periods=1).mean())

    print(f"Executing macro keyword extraction for {len(gold_final)} missing dates...")
    loop = asyncio.get_event_loop()
    try:
        articles_results = loop.run_until_complete(process_all_dates(gold_final["Date"], ALL_KEYWORDS))
    except Exception:
        articles_results = asyncio.run(process_all_dates(gold_final["Date"], ALL_KEYWORDS))

    article1, article2, article3, article4, article5 = [], [], [], [], []
    for date in gold_final["Date"]:
        arts = articles_results[date]
        article1.append(arts[0])
        article2.append(arts[1])
        article3.append(arts[2])
        article4.append(arts[3])
        article5.append(arts[4])

    gold_final["Article1"] = article1
    gold_final["Article2"] = article2
    gold_final["Article3"] = article3
    gold_final["Article4"] = article4
    gold_final["Article5"] = article5

    if not existing.empty:
        df = pd.concat([existing, gold_final], ignore_index=True)
    else:
        df = gold_final

    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.sort_values(by="Date")

    if len(df) > 2500:
        df = df.tail(2500)

    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df.to_sql("gold_prices", conn, if_exists="replace", index=False)
    df.to_csv(CSV_PATH, index=False)
    
    print(f"SQLite Database and CSV completely updated ending dynamically at TODAY -> {df['Date'].iloc[-1]}")
    conn.close()
    return df

if __name__ == "__main__":
    update_dataset()