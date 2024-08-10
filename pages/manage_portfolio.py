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


st.set_page_config(layout="wide", page_title="Manage Your Portfolio", page_icon="📈")
st.write(
    """
# Manage Your Portfolio
"""
)

with st.sidebar:
    st.write("## Portfolio")
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
    st.write("## Observation Window")
    observation_start_date = st.date_input("**Start Date**", datetime.date(2024, 8, 1))
    observation_end_date = st.date_input(
        "**End Date**", datetime.datetime.today().date()
    )
    opt_mode = st.selectbox(
        "**Aim**",
        ["Expected Annual Return Rate (%)", "Maximum Tolerable Risk (%)"],
        index=0,
    )
    expected_return_rate_or_tolerable_risk = st.slider("", 0, 100, 1)
    # tolerable_risk = st.slider("**Maximum Tolerable Risk (%)**", 0, 100, 20)


trading_date = st.date_input("**Trading Date**", datetime.datetime.today().date())
stock_worth = []
col_count = 5
cols = []
for index, stock_code in enumerate(selected_stocks):
    if index % col_count == 0:
        cols = st.columns(col_count)
    with cols[index % col_count]:
        stock_worth.append(st.text_input(stock_code, "0"))

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
        y=normalize_price_df(stock_price[StockMetric().CLOSE]),
        mode="lines",
        name=stock_code,
    )
fig.update_layout(xaxis_title="Date", yaxis_title="Price")

st.plotly_chart(fig, use_container_width=True)


def datetime_to_pd_timestamp(date):
    return pd.Timestamp(year=date.year, month=date.month, day=date.day)


observation_window_prices = []
for stock_code in selected_stocks:
    price = api.get_stock_price(stock_code, selected_duration)
    print(observation_start_date)
    price.index = price.index.tz_localize(None)
    observation_window_prices.append(
        (
            stock_code,
            price[
                (price.index >= datetime_to_pd_timestamp(observation_start_date))
                & (price.index <= datetime_to_pd_timestamp(observation_end_date))
            ],
        )
    )


correlation_col, mean_col, std_col = st.columns([3, 1, 1])

with correlation_col:
    st.header("Correlation")

    def correlation_table(stock_codes, observation_window_prices):
        df_col = pd.DataFrame(
            {
                code: normalize_date_index(price)
                for code, price in zip(stock_codes, observation_window_prices)
            }
        )
        df_corr = df_col.corr()
        st.write(df_corr)
        plt.matshow(df_col.corr())

    correlation_table(
        [price[0] for price in observation_window_prices],
        [
            normalize_price_df(price[1][StockMetric().CLOSE]).diff()
            for price in observation_window_prices
        ],
    )


with mean_col:
    st.header("Mean")

    def mean_table(stock_codes, price_deltas):
        df_col = pd.DataFrame(
            {code: price for code, price in zip(stock_codes, price_deltas)}
        )
        df_mean = df_col.mean()
        st.write(df_mean)

    mean_table(
        [price[0] for price in observation_window_prices],
        [
            normalize_price_df(price[1][StockMetric().CLOSE]).diff()
            for price in observation_window_prices
        ],
    )


with std_col:
    st.header("Std")

    def standard_variance_table(stock_codes, price_deltas):
        df_col = pd.DataFrame(
            {code: price for code, price in zip(stock_codes, price_deltas)}
        )
        df_std = df_col.std()
        st.write(df_std)

    standard_variance_table(
        [price[0] for price in observation_window_prices],
        [
            normalize_price_df(price[1][StockMetric().CLOSE]).diff()
            for price in observation_window_prices
        ],
    )


historical_return_rates = pd.DataFrame(
    {
        price[0]: normalize_date_index(
            normalize_price_df(price[1][StockMetric().CLOSE])
        ).diff()
        for price in observation_window_prices
    }
)
historical_return_rate_mean = historical_return_rates.mean()
historical_return_rate_cov = historical_return_rates.cov()
# target_return_rate = 0.00001  # 目标收益率
weights = np.random.random(len(observation_window_prices))

print(historical_return_rate_mean)
print(historical_return_rate_cov)

bnds = tuple((0, 1) for _ in weights)


def portfolio_return_rate(weights):
    return np.sum(historical_return_rate_mean * weights * 250.0)


def portfolio_std(weights):
    return np.sqrt(
        np.dot(weights.T, np.dot(historical_return_rate_cov, weights)) * 250.0
    )


def negative_portfolio_return_rate(weights):
    return -portfolio_return_rate(weights)


cons = (
    {
        "type": "eq",
        "fun": lambda x: (
            portfolio_return_rate(x) - expected_return_rate_or_tolerable_risk * 0.01
            if opt_mode == 0
            else portfolio_std(x) - expected_return_rate_or_tolerable_risk * 0.01
        ),
    },
    {"type": "eq", "fun": lambda x: np.sum(x) - 1},
)

eweights = np.array(
    len(observation_window_prices)
    * [
        1.0 / len(observation_window_prices),
    ]
)

res = sco.minimize(
    portfolio_std if opt_mode == 0 else negative_portfolio_return_rate,
    eweights,
    method="SLSQP",
    bounds=bnds,
    constraints=cons,
)
print(res)

if res.success == True:
    with np.printoptions(precision=4, suppress=True):
        st.write(str(res.fun))
        st.write(str(res.x))
