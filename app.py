import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import datetime
import matplotlib.pyplot as plt


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

st.set_page_config(layout="wide")
st.write(
    """
# Manage Your Portfolio
"""
)

with st.sidebar:
    selected_stocks = st.multiselect(
        "Choose Stocks",
        ["GOOG", "^SPX", "000001.SS", "^IXIC", "AAPL", "^N225"],
        ["GOOG", "^SPX"],
    )
    selected_duration = st.selectbox(
        "Time Range",
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
for stock_code in selected_stocks:
    prices.append((stock_code, api.get_stock_price(stock_code, selected_duration)))

fig = px.line()

for price in prices:
    stock_code = price[0]
    stock_price = price[1]
    fig.add_scatter(
        x=stock_price.index,
        y=stock_price[StockMetric().CLOSE] * 1.0 / stock_price[StockMetric().CLOSE][0],
        mode="lines",
        name=stock_code,
    )
fig.update_layout(xaxis_title="Date", yaxis_title="Price")

st.plotly_chart(fig, use_container_width=True)


def correlation_table(stock_codes, prices):
    df_col = pd.DataFrame({code: price for code, price in zip(stock_codes, prices)})
    df_corr = df_col.corr()
    st.write(df_corr)
    plt.matshow(df_col.corr())


st.write("## Correlations")

correlation_table(
    [price[0] for price in prices], [price[1][StockMetric().CLOSE] for price in prices]
)
