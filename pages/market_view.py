import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import datetime
import matplotlib.pyplot as plt
import scipy.optimize as sco
import numpy as np


class Duration:
    def __init__(self) -> None:
        self.ONE_DAY = "1d"
        self.FIVE_DAY = "5d"
        self.ONE_MONTH = "1mo"
        self.THREE_MONTH = "3mo"
        self.SIX_MONTH = "6mo"
        self.ONE_YEAR = "1y"
        self.TWO_YEAR = "2y"
        self.FIVE_YEAR = "5y"
        self.TEN_YEAR = "10y"
        self.YTD = "ytd"
        self.MAX = "max"


class StockMetric:
    def __init__(self) -> None:
        self.HIGH = "High"
        self.OPEN = "Open"
        self.LOW = "Low"
        self.CLOSE = "Close"
        self.VOLUME = "Volume"
        self.DIVIDENDS = "Dividends"
        self.STOCK_SPLITS = "Stock Splits"


class YahooFinanceApi:
    def __init__(self) -> None:
        pass

    def get_stock_price(self, stock_code: str, duration: str):
        stock = yf.Ticker(stock_code)
        hist = stock.history(period=duration)
        return hist

    def get_options(self, stock_code):
        stock = yf.Ticker(stock_code)

        # Get options for each expiration
        options = pd.DataFrame()
        for e in stock.options:
            opt = stock.option_chain(e)
            opt = pd.concat([opt.calls, opt.puts], ignore_index=True)
            opt["expirationDate"] = e
            options = pd.concat([options, opt], ignore_index=True)

        # Bizarre error in yfinance that gives the wrong expiration date
        # Add 1 day to get the correct expiration date
        options["expirationDate"] = pd.to_datetime(
            options["expirationDate"]
        ) + datetime.timedelta(days=1)
        options["dte"] = (
            options["expirationDate"] - datetime.datetime.today()
        ).dt.days / 365

        # Boolean column if the option is a CALL
        options["CALL"] = options["contractSymbol"].str[4:].apply(lambda x: "C" in x)

        options[["bid", "ask", "strike"]] = options[["bid", "ask", "strike"]].apply(
            pd.to_numeric
        )
        options["mark"] = (
            options["bid"] + options["ask"]
        ) / 2  # Calculate the midpoint of the bid-ask

        # Drop unnecessary and meaningless columns
        options = options.drop(
            columns=[
                "contractSize",
                "currency",
                "change",
                "percentChange",
                "lastTradeDate",
                "lastPrice",
            ]
        )
        return options


# tickers = yf.Tickers("MSFT GOOG AAPL")

# # get all stock info
# print(tickers.tickers["MSFT"].info)
# print(tickers.tickers["GOOG"].info)
# print(tickers.tickers["AAPL"].info)

# # get historical market data
# hist = msft.history(period="max")

# # show meta information about the history (requires history() to be called first)
# msft.history_metadata

# # show actions (dividends, splits, capital gains)
# msft.actions
# msft.dividends
# msft.splits
# msft.capital_gains  # only for mutual funds & etfs

# print(msft.dividends)
# print(msft.splits)

# # show share count
# msft.get_shares_full(start="2022-01-01", end=None)

# # show financials:
# # - income statement
# msft.income_stmt
# msft.quarterly_income_stmt
# # - balance sheet
# msft.balance_sheet
# msft.quarterly_balance_sheet
# # - cash flow statement
# msft.cashflow
# msft.quarterly_cashflow
# # see `Ticker.get_income_stmt()` for more options

# print(msft.balance_sheet)

# # show holders
# msft.major_holders
# msft.institutional_holders
# msft.mutualfund_holders
# msft.insider_transactions
# msft.insider_purchases
# msft.insider_roster_holders

# msft.sustainability

# # show recommendations
# msft.recommendations
# msft.recommendations_summary
# msft.upgrades_downgrades

# # Show future and historic earnings dates, returns at most next 4 quarters and last 8 quarters by default.
# # Note: If more are needed use msft.get_earnings_dates(limit=XX) with increased limit argument.
# msft.earnings_dates

# # show ISIN code - *experimental*
# # ISIN = International Securities Identification Number
# msft.isin

# # show options expirations
# msft.options

# # show news
# msft.news

# print(msft.news)

# # get option chain for specific expiration
# # opt = msft.option_chain('YYYY-MM-DD')
# # data available via: opt.calls, opt.puts


def normalize_price_df(df):
    return df / df[0]


def normalize_date_index(df):
    df.index = df.index.date
    return df


st.set_page_config(layout="wide", page_title="Market View", page_icon="📊")
st.write(
    """
# Market View
"""
)

with st.sidebar:
    selected_stocks = st.multiselect(
        "**Choose Stocks**",
        [
            "GOOG",
            "^SPX",
            "000001.SS",
            "^IXIC",
            "AAPL",
            "^N225",
            "NVDA",
            "^HSI",
            "MSFT",
            "META",
            "TSLA",
            "AMZN",
            "601988.SS",
        ],
        ["GOOG", "^SPX"],
    )
    selected_duration = st.selectbox(
        "**Time Range**",
        [
            Duration().ONE_DAY,
            Duration().FIVE_DAY,
            Duration().ONE_MONTH,
            Duration().THREE_MONTH,
            Duration().SIX_MONTH,
            Duration().ONE_YEAR,
            Duration().TWO_YEAR,
            Duration().FIVE_YEAR,
            Duration().TEN_YEAR,
            Duration().MAX,
        ],
        index=1,
    )


api = YahooFinanceApi()
prices = []
options = []
for stock_code in selected_stocks:
    prices.append((stock_code, api.get_stock_price(stock_code, selected_duration)))
    options.append(api.get_options(stock_code))


st.write("# Price")

fig = px.line()

for price in prices:
    stock_code = price[0]
    stock_price = price[1]
    fig.add_scatter(
        x=stock_price.index,
        y=normalize_price_df(stock_price[StockMetric().CLOSE]),
        mode="lines",
        name=stock_code,
    )
fig.update_layout(xaxis_title="Date", yaxis_title="Price")

st.plotly_chart(fig, use_container_width=True)

st.write("# Options")

for stock_code, option in zip(selected_stocks, options):
    fig = px.line()
    for expiringDate in option["expirationDate"].unique():
        if expiringDate > pd.Timestamp.today() + pd.Timedelta(days=30 * 1):
            continue
        expiring_date_filter = option["expirationDate"] == expiringDate
        call_filter = option["CALL"] == True
        filtered_option = option.where(expiring_date_filter)
        filtered_option = filtered_option.where(call_filter)
        fig.add_scatter(
            x=filtered_option["strike"],
            y=filtered_option["impliedVolatility"],
            mode="lines",
            name="call-" + str(expiringDate),
        )
        call_filter = option["CALL"] == False
        filtered_option = option.where(expiring_date_filter)
        filtered_option = filtered_option.where(call_filter)
        fig.add_scatter(
            x=filtered_option["strike"],
            y=filtered_option["impliedVolatility"],
            mode="lines",
            name="put-" + str(expiringDate),
        )
    fig.update_layout(
        title=stock_code, xaxis_title="Strike Price", yaxis_title="Implicit Volatility"
    )
    st.plotly_chart(fig, use_container_width=True)
    # st.write(option)
