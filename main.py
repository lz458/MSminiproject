import streamlit as st
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from dashboard import *
import os
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd


#********************main page setting********************
st.set_page_config(layout="wide")
st.title('Chris Zhang MS Mini Project')
page = st.sidebar.selectbox(
    'Select Page',
    ['Home', 'Relative Value Analysis',  'Volatility Analysis', 'Rates Analysis', 'Portfolio Analysis, Risk Management & Hedging']
)

#********************Home page setting********************

if page == 'Home':
    st.header('Home')
    selected_horizon = st.sidebar.selectbox(
        'Select horizon',
        ["1y", "5y","1mo","3mo","6mo"],
        index=1
    )
    st.sidebar.subheader("Positions in thousands")

    tsla_po = st.sidebar.number_input('TSLA Position', value=10)
    spx_po = st.sidebar.number_input('SPX Position', value=80)
    go_po = st.sidebar.number_input('Gold Position', value=10)

    #***********************Horizon Settings and positions initialisation done ******************
    st.subheader("Current Market Price")
    st.write(f"Dial range is the {selected_horizon} Min/Max range, green bar is the past week range")
    col1, col2, col3 = st.columns(3)
    with col1:
        tesla = Home("TSLA", selected_horizon)
        tesla_plot = tesla.dials()
        st.plotly_chart(tesla_plot)
    with col2:
        spx = Home("^GSPC", selected_horizon)
        spx_plot = spx.dials()
        st.plotly_chart(spx_plot)
    with col3:
        gold = Home("GC=F", selected_horizon)
        gold_plot = gold.dials()
        st.plotly_chart(gold_plot)
    
    #*************Dial Plotting done, start calculating position market values**************
    st.subheader("Market Value Changes")
    col1, col2 = st.columns(2)
    positions = {
        "TSLA": tsla_po,
        "^GSPC": spx_po,
        "GC=F": go_po
    }
    positionsList = ['TSLA', "GC=F", "^GSPC"]
    dataall = download_data(positionsList, selected_horizon)
    #doesn't matter which instance to use to plot, methods are the same and data are the same. my clumsy structuring, later pages only need to create 1 instance of the class
    market_values = tesla.calculate_market_value(dataall, positions)
    #*************Plotting market value and chart market value 
    with col1:
        for asset in market_values.columns:
            initial_value = market_values[asset].iloc[0]
            final_value = market_values[asset].iloc[-1]
            delta = final_value - initial_value
            percentage_change = (delta / initial_value) * 100
            num_days = len(market_values)
            trading_days_per_year = 252 
            annualised_percentage_change = ((1 + percentage_change / 100) ** (trading_days_per_year / num_days) - 1) * 100
            st.metric(
                label=f"{asset} Market Value Change", 
                value=f"${round(final_value,2)}", 
                delta=f"{round(delta,2)} (${round(annualised_percentage_change,2)}% Annualised)" #Show in annualised percentage terms and only keey 2 decimal places
            )
    with col2:
        st.pyplot(tesla.plot_time_series_with_multiple_y_axes(market_values))
    #*************Plotting the total  portfolio performance 
    st.subheader("Overall Portfolio Performance")
    overall_performance = market_values.sum(axis=1) #sum on axis 1 horizontally to get total portfolio value
    initial_overall_value = overall_performance.iloc[0]
    final_overall_value = overall_performance.iloc[-1]
    overall_delta = final_overall_value - initial_overall_value
    overall_percentage_change = (overall_delta / initial_overall_value) * 100
    overall_annualised_percentage_change = ((1 + overall_percentage_change / 100) ** (trading_days_per_year / num_days) - 1) * 100

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Overall Portfolio Value Change", 
            value=f"${round(final_overall_value,2)}", 
            delta=f"${round(overall_delta,2)} ({round(overall_annualised_percentage_change,2)}% Annualised)"
        )

    with col2:
        st.pyplot(tesla.plot_cumulative_sum(market_values))

#------------------------Market data only analysis------------
   
    st.subheader("Time Series Data Plot")
    selected_plot = st.selectbox(
        'Select Plot to Display',
        ['Tesla Time Series', 'SPX Time Series', 'Gold Time Series']
    )

    if selected_plot == 'Tesla Time Series':
        st.pyplot(tesla.timeseires())
    elif selected_plot == 'SPX Time Series':
        st.pyplot(spx.timeseires())
    elif selected_plot == 'Gold Time Series':
        st.pyplot(gold.timeseires())
    col1, col2 = st.columns(2)
    st.subheader("Combined Time Series Data Plot")

    #*************Normalised plot
    with col1:
        normalise = st.checkbox('Normalise Time Series', value=True)
    with col2:
        selected_variables = st.multiselect(
            'Select Variables to Use',
            ['TSLA', "GC=F", "^GSPC"],
            default=['TSLA', "GC=F", "^GSPC"]
        )
        if len(selected_variables) <= 1:
            st.error("Please select more than one variable.")

    data = download_data(selected_variables, selected_horizon)
    if normalise:
        time_series_plot = tesla.plot_normalised_time_series(data)
    else:
        time_series_plot = tesla.plot_time_series_with_multiple_y_axes(data)
    st.pyplot(time_series_plot)
    #*************Data Download
    st.subheader("Downloaded Data")
    col1, col2 = st.columns(2)

    with col1:
        st.write("downloaded market data")
        st.dataframe(dataall)

    with col2:
        st.write("Market Values Data")
        st.dataframe(market_values)

elif page == 'Relative Value Analysis':

    st.header('PCA Analysis')
    st.write("PCA residual time seires are plotted using static PCA, i.e., the algorithm only ran once. For more acurate result. dynamic PCA, taking the latest residual and form a time series can be considered. Results are not significantly different in real application")
    selected_target = st.sidebar.selectbox(
        'Select Target',
        ["TSLA", "^SPX", "GC=F"],
        index=0
    )
    if selected_target == "TSLA":
        company_options = ["AAPL", "TSLA", "GOOG", "MSFT", "AMZN","META","NVDA"]
    elif selected_target == "^SPX":
        company_options = ["^SPX", "^IXIC", "^FTSE", "^STOXX50E", "^HSI"]
    elif selected_target == "GC=F":
        company_options = ["GC=F", "SI=F", "PL=F", "GDX","HG=F"]
    else:
        company_options = []
    selected_comps = st.sidebar.multiselect(
        'Select Comps',
        company_options,
        default=company_options[:5]  # Default to the first 5 companies
    )
    if len(selected_comps) < 4:
        st.error("Please select at least 4 companies.")
    selected_horizon = st.sidebar.selectbox(
        'Select Horizon',
        ["1y", "5y","3mo","6mo"],
        index=1
    )
    #*************setting complete, calling function to run analysis 
    if st.sidebar.button('Run Analysis'):
        rv = pcaAnalysis(selected_target,selected_comps, selected_horizon)
        pca_residuals, pca_z_scores, cumulative_explained_variance  = rv.pca_analysis() #Getting all the function results 
        pca_z_scores = pd.DataFrame(pca_z_scores, columns=rv.data.columns, index=rv.data.index)
        col1,col2 = st.columns(2)
        with col1:
            st.metric(
                label="PCA residual Z score Change (Past Week) Higher the Richer",
                value=f"{round(pca_z_scores[f'{selected_target}'][-1].item(),2)}",
                delta=f"{round((round(pca_z_scores[f'{selected_target}'][-1],2) - round(pca_z_scores[f'{selected_target}'][-6],2)).item(),2)}"
            )
        with col2:
            st.plotly_chart(rv.dials(pca_z_scores))
        #****************************************************Validity check
        st.subheader("Validity check")
        col1,col2 = st.columns(2)
        with col1:        
            st.pyplot(rv.plt_cumulative_explained_variance(cumulative_explained_variance))
        with col2:
            st.write("Should expect high variance explained ratio using 3 PCs, if not, the fowlling analysis validity will be reduced")
        #****************************************************Residual analysis     
        st.subheader("PCA residual analysis")
        col1,col2 = st.columns(2)
        with col1:
            if selected_horizon == "3mo":
                st.write("Selected time frame too short for time series")
            else:
                st.pyplot(rv.plt_moving_avg_residuals(pca_z_scores))
        with col2:
            st.pyplot(rv.plt_latest_residuals(pca_z_scores))
        #****************************************************Fundamental analysis     
        if selected_target == "TSLA":
            st.subheader("Fundamental analysis")
            st.pyplot(rv.plot_pe_ratios(rv.comps))
        else:
            st.subheader("Fundamental analysis")
            st.write("Fundamental analysis only available for single name stocks")
        #****************************************************Data download    
        st.subheader(f'{selected_target} Data')
        rv.data.index = rv.data.index.strftime('%d/%m/%Y')# Format the index to display as dd/mm/yyyy
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f'{selected_target} Data')
            st.dataframe(rv.data)
        with col2:
            st.subheader('PCA Residuals Z-Scores')
            st.dataframe(pca_z_scores)

elif page == 'Volatility Analysis':
    st.header('Volatility Analysis')
    selected_target = st.sidebar.selectbox(
        'Select Target',
        ["TSLA", "^SPX", "GC=F"],
        index=0
    )
    selected_horizon = st.sidebar.selectbox(
        'Select horizon',
        ["1y", "5y","1mo","3mo","6mo"],
        index=1
    )
    riskthreshold = st.sidebar.number_input('Risk threshold', value=10)
    riskfree_Rate = st.sidebar.number_input('Risk Free Rate', value=4)
    riskfree_Rate = riskfree_Rate/100
    confidence_interval = st.sidebar.number_input('Confidence Interval', value=95)
    confidence_interval = confidence_interval/100
    analyse = st.sidebar.button("Analyse")
    page = volitility()
    #********************************************Init complete 
    if analyse:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Daily price change plot")
            st.pyplot(page.plot_daily_pct_change_with_highlight(download_data(selected_target, selected_horizon), riskthreshold, -riskthreshold))
        with col2:
            st.subheader("Volatility Data")
            st.dataframe(page.calculate_statistics(selected_target,download_data(selected_target, selected_horizon), riskfree_Rate))
        with col3:
            st.subheader("VaR analysis")
            var = page.calculate_var(download_data(selected_target, selected_horizon),confidence_interval)
            st.write(f"There is {confidence_interval*100}% change that the position loss will not exceed {-var *100}%")

        #********************************************More plots
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(page.volHistogram(selected_target, selected_horizon))

        with col2:
            st.pyplot(page.plot_volatility_barchart(download_data(selected_target, "5y")))
        #********************************************Rolling
        if selected_horizon in ["1y", "5y"]:
            st.subheader("30 day rolling Volatility")
            st.pyplot(page.plot_30_day_rolling_average_volatility(download_data(selected_target, selected_horizon)))
            st.subheader("30 day rolling Sharpe ratio, geometric")
            st.pyplot(page.plot_90_day_rolling_sharpe_ratio(download_data(selected_target, selected_horizon), riskfree_Rate))
        else:
            st.subheader("30 day rolling Volatility")
            st.write("Volitility time series analysis not available, time series is too short")

elif page == 'Rates Analysis':
    st.header('Rates Analysis')    
    rates = Rates()
    selected_rate = st.sidebar.selectbox(
        'Select Rate',
        ["2y", "5y", "10y", "30y", "2s10s", "5s30s"],
        index=0
    )
    selected_target = st.sidebar.selectbox(
        'Select Target',
        ["TSLA", "^SPX", "GC=F"],
        index=0
    )
    selected_horizon = st.sidebar.selectbox(
        'Select horizon',
        ["1y", "5y", "5d", "1mo", "3mo", "6mo"],
        index=1
    )
    roll = st.sidebar.number_input('Numbers of days for rolling analysis', value=60)
    st.sidebar.subheader("Postions set up")
    tsla_po = st.sidebar.number_input('TSLA Position', value=10)
    spx_po = st.sidebar.number_input('SPX Position', value=80)
    go_po = st.sidebar.number_input('Gold Position', value=10)
    
    #********************************************Init complete 
    if st.sidebar.button('Run Rates Analysis'):
        rates_data = rates.data(selected_horizon)
        st.subheader("Rates sensitivity against the selected position")
        st.plotly_chart(rates.plot_rolling_correlation(rates_data[f"{selected_rate}"], download_data(selected_target, selected_horizon), roll))
        
        positions = {
        "TSLA": tsla_po,
        "^GSPC": spx_po,
        "GC=F": go_po
        }
        positionsList = ['TSLA', "GC=F", "^GSPC"]
        dataall = download_data(positionsList, selected_horizon)
        market_values = rates.calculate_market_value(dataall, positions)
        st.subheader("Rates sensitivity against overall portfolio")
        st.plotly_chart(rates.plot_rolling_correlation(rates_data[f"{selected_rate}"],market_values["sum"],roll))
        st.pyplot(rates.plot_correlation_matrix(rates.data(selected_horizon)))
        st.subheader("Rates Data")
        st.dataframe(rates_data)

elif page == 'Portfolio Analysis, Risk Management & Hedging':
    st.header('Portfolio Analysis, Risk Management & Hedging')
    selected_horizon = st.sidebar.selectbox(
        'Select horizon',
        ["1y", "5y", "3mo", "6mo"],
        index=1
    )
    tsla_weight = st.sidebar.number_input('TSLA Weight', value=0.1, min_value=0.0, max_value=1.0, step=0.01)
    spx_weight = st.sidebar.number_input('SPX Weight', value=0.8, min_value=0.0, max_value=1.0, step=0.01)
    gold_weight = st.sidebar.number_input('Gold Weight', value=0.1, min_value=0.0, max_value=1.0, step=0.01)
    total_weight = tsla_weight + spx_weight + gold_weight

    unhedgedPortfolio= {
    "GC=F": gold_weight,
    "TSLA":tsla_weight,
    "^SPX": spx_weight,
    }
    if total_weight != 1.0:
        st.error(f"Total weight must be 1.0, but it is currently {total_weight:.2f}")
    st.sidebar.subheader("Risk Free Rate")
    Rf = st.sidebar.number_input("", value=1.0, min_value=-5.0, max_value=5.0, step=0.25)
    Rf = Rf/100
    st.sidebar.subheader("Hedging Options")
    Treasuries = st.sidebar.checkbox('Treasuries', value=True)
    Silver = st.sidebar.checkbox('Silver', value=True)
    NASDAQ = st.sidebar.checkbox('NASDAQ', value=True)
    retail = st.sidebar.checkbox('Walmart', value=True)
    oil = st.sidebar.checkbox('Exxon Mobil', value=True)


    #********************************************Init complete 

    Analyse = st.sidebar.button("Analyse")

    if Analyse:
        portfolio = Portfolio()
        data = download_data(["TSLA", "^SPX", "GC=F"], selected_horizon)
        optimised_portfolio = portfolio.mean_variance_optimised_portfolio(data,Rf)
        optimised_portfolio_plot = optimised_portfolio
        optimised_portfolio = {k: f"{v:.2f}" for k, v in optimised_portfolio.items()} #Data time will be converted to string, becareful calling in plots or df 
        #********************************************Unhedged portfolio optimised 
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Optimised Portfolio")
            st.write(optimised_portfolio)
        with col2:
            st.subheader("Current Portfolio Weights")
            st. write(unhedgedPortfolio)
        #********************************************Finished writing the optimised port 
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(portfolio.plot_comparison_bar_chart_position(optimised_portfolio,unhedgedPortfolio))
        with col2:
            st.pyplot(portfolio.plot_efficient_frontier(data, np.array([tsla_weight, spx_weight, gold_weight]), 
                                                        np.array(list(optimised_portfolio_plot.values())), 
                                                        Rf))

        #********************************************Start hedging 

        st.header("Implementing hedging")
        st.subheader("Opmised Hedged Portfolio")
        hedgePorfolio = {"TSLA": tsla_weight, "^SPX": spx_weight, "GC=F": gold_weight}
        if Treasuries:
            hedgePorfolio["^TNX"] = 0
        if Silver:
            hedgePorfolio["SI=F"] = 0
        if NASDAQ:
            hedgePorfolio["^IXIC"] = 0
        if oil:
            hedgePorfolio["XOM"] = 0
        if retail:
            hedgePorfolio["WMT"] = 0
        data_Hedged = download_data(list(hedgePorfolio.keys()),selected_horizon)
        hedgePorfolio = portfolio.mean_variance_optimised_portfolio_with_constraints(data_Hedged,hedgePorfolio,Rf)
        hedgePorfolio = {k: round(v, 2) for k, v in hedgePorfolio.items()}
        st.write(hedgePorfolio)
        #********************************************Finished optimising the optimal port 

        marketValue = portfolio.calculate_market_value(data,unhedgedPortfolio)
        hedgedMarketValue = portfolio.calculate_market_value(data_Hedged,hedgePorfolio)#Portfolio market value

        results = portfolio.calculate_portfolio_metrics(marketValue,Rf)
        results_hedged = portfolio.calculate_portfolio_metrics(hedgedMarketValue,Rf)

      

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Unhedged Portfolio Metrics")
            st.write(results)

        with col2:
            st.subheader("Hedged Portfolio Metrics")
            st.write(results_hedged)


        st.pyplot(portfolio.plot_comparison_bar_chart_metrics(results_hedged,results))

        st.pyplot(portfolio.plot_cumulative_sum(marketValue,hedgedMarketValue))

        st.pyplot(portfolio.plot_histogram_with_normal_fit(marketValue, hedgedMarketValue))
