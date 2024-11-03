import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from scipy.optimize import minimize
from scipy.stats import norm
import plotly.graph_objects as go


positions ={
    "TSLA": 10000,
    "SPX": 80000,
    "Gold": 10000
}


def download_data(ticker, horizon):
    stock = yf.download(ticker,period=horizon)
    data = stock["Close"]
    return data

class pcaAnalysis:
    def __init__(self, target, comps,horizon):
        self.horizon = horizon
        self.comps = comps
        self.target = target
        self.data = download_data(self.comps, horizon)
        self.data = self.data.dropna()

    def pca_analysis(self):
        scaler = StandardScaler()
        pca = PCA(n_components=3)
        scaled_data = scaler.fit_transform(self.data)
        pca.fit(scaled_data)
        projected_data = pca.transform(scaled_data)
        reconstructed_data = pca.inverse_transform(projected_data)
        pca_residuals = scaled_data - reconstructed_data
        pca_z_scores = zscore(pca_residuals)
        cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
        return pca_residuals, pca_z_scores, cumulative_explained_variance

    def plt_cumulative_explained_variance(self, cumulative_explained_variance):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--', color='b')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Cumulative Explained Variance Ratio')
        ax.set_title('Cumulative Explained Variance by Principal Components')
        ax.set_xticks(range(1, len(cumulative_explained_variance) + 1))
        ax.grid(True)
        return fig

    def plt_moving_avg_residuals(self, pca_z_scores):
        #hard coded for 60 days
        moving_avg_z_scores = pd.DataFrame(pca_z_scores, index=self.data.index, columns=self.data.columns).rolling(window=60).mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(moving_avg_z_scores)
        ax.set_xlabel('Date')
        ax.set_ylabel('60-Day Moving Average of Residuals')
        ax.set_title('60-Day Moving Average of PCA Residuals Over Time')
        ax.legend(self.data.columns)
        return fig

    def plot_pe_ratios(self,tickers):
        pe_ratios = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            pe_ratio = stock.info.get('trailingPE', None)
            if pe_ratio is not None:
                pe_ratios[ticker] = pe_ratio

        if not pe_ratios:
            print("No P/E ratios found for the given tickers.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(pe_ratios.keys(), pe_ratios.values(), color='blue', edgecolor='black')
        ax.set_title('P/E Ratios of Selected Tickers')
        ax.set_xlabel('Ticker')
        ax.set_ylabel('P/E Ratio')
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    def plt_latest_residuals(self, pca_z_scores):
        fig, ax = plt.subplots(figsize=(10, 6))
        latest_residuals = pca_z_scores.iloc[-1] 
        ax.bar(latest_residuals.index, latest_residuals.values, alpha=0.7)
        ax.set_xlabel('Component')
        ax.set_ylabel('Residuals')
        ax.set_title('Latest PCA Residuals (Bar Chart)')
        return fig

    def dials(self,pca_z_scores):
        data = pca_z_scores[self.target]
        current_value = data.iloc[-1]
        min_value = data.min()
        max_value = data.max()
        # Calculate the past month's range
        one_month_ago = data.index[-1] - pd.DateOffset(months=1)
        past_month_data = data[data.index >= one_month_ago]
        past_month_min = past_month_data.min()
        past_month_max = past_month_data.max()
        mid_point = (min_value + max_value) / 2
        data_past_week = data.iloc[-5]
        #-------------------------------------
        # Create the gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            delta={
                'reference': data_past_week,  # Plotting change against last week 
                'increasing': {'color': "red"},
                'decreasing': {'color': "green"}
            },
            gauge={
                'axis': {'range': [min_value, max_value], 'tickwidth': 1, 'tickcolor': "black"},
                'bgcolor': "rgba(0, 0, 0, 0)",  # Transparent background
                'borderwidth': 2,
                'bordercolor': "gray",
                'bar': {'color': "rgba(255, 255, 255, 0)"},
                'steps': [
                    {'range': [min_value, mid_point], 'color': "Green"},
                    {'range': [mid_point, max_value], 'color': "Red"},
                    {'range': [past_month_min, past_month_max], 'color': "rgb(66, 20, 95)"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': current_value
                }
            }
        ))

        # title anno
        fig.add_annotation(
            text=f"{self.target}",
            x=0.5,
            y=1,
            showarrow=False,
            font=dict(size=15, color="white"),
            bgcolor="rgb(66, 20, 95)",
            bordercolor="rgb(66, 20, 95)",
            borderwidth=2,
            borderpad=4,
            width=200,
            xref="paper",
            yref="paper"
        )

        # min max anno
        fig.add_annotation(
            text=f"Rich: {round(max_value,2)}",
            x=1.05,
            y=0.1,
            showarrow=False,
            font=dict(size=14, color="Red"),
            bgcolor="rgba(255, 255, 255, 0)",  # Transparent annotation background
            bordercolor="rgba(255, 255, 255, 0)",
            borderwidth=1,
            borderpad=4,
        )

        fig.add_annotation(
            text=f"Cheap: {round(min_value,2)}",
            x=-0.03,
            y=0.1,
            showarrow=False,
            font=dict(size=14, color="Green"),
            bgcolor="rgba(255, 255, 255, 0)",  # Transparent annotation background
            bordercolor="rgba(255, 255, 255, 0)",
            borderwidth=1,
            borderpad=4,
        )

        # Size and Margin
        fig.update_layout(
            paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent overall background
            font={'color': "black", 'family': "Arial"},
            width=300,  # Adjusted width
            height=300, # Adjusted height
            margin=dict(t=20, b=10, l=30, r=30)  # Adjust margins to fit title
        )

        return fig

class Home:

    def __init__(self, target,horizon):
        self.horizon = horizon
        
        self.target = target
        self.data = download_data(self.target, horizon)
    
    def dials(self):
        current_value = self.data.iloc[-1]
        min_value = self.data.min()
        max_value = self.data.max()

        # Calculate the past month's range
        one_month_ago = self.data.index[-1] - pd.DateOffset(months=1)
        past_month_data = self.data[self.data.index >= one_month_ago]
        past_month_min = past_month_data.min()
        past_month_max = past_month_data.max()
        mid_point = (min_value + max_value) / 2
        data_past_week = self.data.iloc[-5]  # Get the specific data point from 5 days ago
        #----------------------------
        # Create the gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            delta={
                'reference': data_past_week,  # past week ref
                'increasing': {'color': "red"},
                'decreasing': {'color': "green"}
            },
            gauge={
                'axis': {'range': [min_value, max_value], 'tickwidth': 1, 'tickcolor': "black"},
                'bgcolor': "rgba(0, 0, 0, 0)",  # Transparent background
                'borderwidth': 2,
                'bordercolor': "gray",
                'bar': {'color': "rgba(255, 255, 255, 0)"},
                'steps': [
                    {'range': [min_value, mid_point], 'color': "Green"},
                    {'range': [mid_point, max_value], 'color': "Red"},
                    {'range': [past_month_min, past_month_max], 'color': "rgb(66, 20, 95)"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': current_value
                }
            }
        ))
        #------------------------------------
        # Add a title annotation centered at the top
        fig.add_annotation(
            text=f"{self.target}",
            x=0.5,
            y=1,
            showarrow=False,
            font=dict(size=15, color="white"),
            bgcolor="rgb(66, 20, 95)",
            bordercolor="rgb(66, 20, 95)",
            borderwidth=2,
            borderpad=4,
            width=200,
            xref="paper",
            yref="paper"
        )

        # Add text annotations for max and min values
        fig.add_annotation(
            text=f"Max: {round(max_value,2)}",
            x=1.05,
            y=0.1,
            showarrow=False,
            font=dict(size=14, color="Red"),
            bgcolor="rgba(255, 255, 255, 0)",  # Transparent annotation background
            bordercolor="rgba(255, 255, 255, 0)",
            borderwidth=1,
            borderpad=4,
        )

        fig.add_annotation(
            text=f"Min: {round(min_value,2)}",
            x=-0.03,
            y=0.1,
            showarrow=False,
            font=dict(size=14, color="Green"),
            bgcolor="rgba(255, 255, 255, 0)",  # Transparent annotation background
            bordercolor="rgba(255, 255, 255, 0)",
            borderwidth=1,
            borderpad=4,
        )

        # Update layout settings including size and margins
        fig.update_layout(
            paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent overall background
            font={'color': "black", 'family': "Arial"},
            width=300,  # Adjusted width
            height=300, # Adjusted height
            margin=dict(t=20, b=10, l=30, r=30)  # Adjust margins to fit title
        )

        return fig

    def timeseires(self):
        fig, ax = plt.subplots()
        ax.plot(self.data.index, self.data, color="blue")
        ax.set_title("Time series price plot")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        max_value = self.data.max()
        min_value = self.data.min()
        max_date = self.data.idxmax()
        min_date = self.data.idxmin()

        #----------------------annot
        ax.plot(max_date, max_value, 'go')  # Green dot for max
        ax.text(max_date, max_value, f'Max: {max_value:.2f}', fontsize=9, color='green', verticalalignment='bottom')
        ax.plot(min_date, min_value, 'ro')  # Red dot for min
        ax.text(min_date, min_value, f'Min: {min_value:.2f}', fontsize=9, color='red', verticalalignment='top')
        return fig
    
    def plot_time_series_with_multiple_y_axes(self, data):
        fig, ax1 = plt.subplots()
        # ------------Plot the first time series
        ax1.plot(data.index, data.iloc[:, 0], label=data.columns[0], color="blue")
        ax1.set_xlabel("Date")
        ax1.set_ylabel(data.columns[0], color="blue")
        ax1.tick_params(axis='y', labelcolor="blue")
        # Create additional y-axes for the remaining time series
        axes = [ax1]
        colors = ["blue", "green", "red", "purple"]
        for i in range(1, data.shape[1]):
            ax = ax1.twinx()
            ax.spines['right'].set_position(('outward', 60 * (i - 1)))  # Offset each additional y-axis, 
            #Each subsequent axis is offset by 60 points multiplied by its index

            ax.plot(data.index, data.iloc[:, i], label=data.columns[i], color=colors[i % len(colors)])
            ax.set_ylabel(data.columns[i], color=colors[i % len(colors)])
            ax.tick_params(axis='y', labelcolor=colors[i % len(colors)])
            axes.append(ax)
            max_date, max_value = data.iloc[:, i].idxmax(), data.iloc[:, i].max()
            min_date, min_value = data.iloc[:, i].idxmin(), data.iloc[:, i].min()
            ax.plot(max_date, max_value, 'go')  # Green dot for max
            ax.plot(min_date, min_value, 'ro')  # Red dot for min
        ax1.set_title("Price history for all securities")
        for i, ax in enumerate(axes):
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1 - 0.1 * i))
        return fig
    
    def plot_normalised_time_series(self, data):
        fig, ax = plt.subplots()
        data_normalised = data / data.iloc[0]
        for column in data_normalised.columns:
            ax.plot(data_normalised.index, data_normalised[column], label=column)
            max_date, max_value = data_normalised[column].idxmax(), data_normalised[column].max()
            min_date, min_value = data_normalised[column].idxmin(), data_normalised[column].min()
            ax.plot(max_date, max_value, 'go')  
            ax.plot(min_date, min_value, 'ro')  
        ax.set_title("Normalised Time Series Plot")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalised Value")
        ax.legend()
        return fig

    def calculate_market_value(self, time_series_data, investments):
        market_values = pd.DataFrame(index=time_series_data.index) #make sure the 2 df will have same index 
        time_series_data = time_series_data.dropna()
        for security, investment in investments.items():
            if security in time_series_data.columns:
                # Calculate the number of shares purchased for each security
                shares = investment / time_series_data[security].iloc[0]
                # Calculate the market value over time
                market_values[security] = shares * time_series_data[security]
            else:
                market_values[security] = np.nan
        market_values = market_values.dropna()
        return market_values

    def plot_cumulative_sum(self, data):
        data = data.dropna()
        cumulative_sum = data.sum(axis=1).to_frame(name='Cumulative Sum') #Summing on axis 1 to get cross sectional sum 
        fig, ax = plt.subplots()
        ax.plot(cumulative_sum.index, cumulative_sum['Cumulative Sum'], label='Cumulative Sum')
        max_date, max_value = cumulative_sum['Cumulative Sum'].idxmax(), cumulative_sum['Cumulative Sum'].max()
        min_date, min_value = cumulative_sum['Cumulative Sum'].idxmin(), cumulative_sum['Cumulative Sum'].min()
        ax.plot(max_date, max_value, 'go')  
        ax.plot(min_date, min_value, 'ro')  
        ax.set_title("Market value")
        ax.set_xlabel("Date")
        ax.set_ylabel("Market value")
        ax.legend()

        return fig

class volitility:

    def volHistogram(self,ticker, horizon):
        data = download_data(ticker,horizon)
        pct_change = data.pct_change().dropna() * 100  # Convert to percentage
        
        # Plot the histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(pct_change, bins=50, color='blue', edgecolor='black')
        ax.set_title(f'Percentage Price Change Histogram for {ticker}')
        ax.set_xlabel('Percentage Change')
        ax.set_ylabel('Frequency')
        ax.grid(True)
        return fig
    
    def calculate_var(self, data, confidence_level):
        daily_returns = data.pct_change().dropna()
        var = np.percentile(daily_returns, (1 - confidence_level) * 100) #based on the normal distribution, if we take the first 5% of the distribution, we get value fo 95% conf in for the VaR
        return round(var, 2)

    def plot_volatility_barchart(self, data):
        # Calculate weekly, monthly, and annual volatility for the past week, month, and year and then annualise it 
        weekly_volatility = data.pct_change().loc[data.index[-7]:].std() * np.sqrt(52)
        monthly_volatility = data.pct_change().loc[data.index[-30]:].std() * np.sqrt(12)
        annual_volatility = data.pct_change().loc[data.index[-365]:].std() * np.sqrt(1)

        # Create a DataFrame to hold the volatilities
        vol_df = pd.DataFrame({
            'Volatility': [weekly_volatility, monthly_volatility, annual_volatility]
        }, index=['Weekly', 'Monthly', 'Annual'])

        # Plot the volatilities as a horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        vol_df.plot(kind='barh', ax=ax) #plotting horizontally 
        ax.set_title('Volatility Bar Chart annualised')
        ax.set_xlabel('Volatility')
        ax.set_ylabel('Period')
        ax.grid(True)
        plt.tight_layout()

        return fig
    

    def plot_30_day_rolling_average_volatility(self, data):
        #30 day is hard coded for now 
        daily_pct_change = data.pct_change().dropna()
        rolling_avg_volatility = daily_pct_change.rolling(window=30).std()

        #--------------plotting--------------------------
        fig, ax = plt.subplots(figsize=(15, 8))  # Make the graph wider
        ax.plot(rolling_avg_volatility.index, rolling_avg_volatility, label='30-Day Rolling Avg Volatility')

        max_date, max_value = rolling_avg_volatility.idxmax(), rolling_avg_volatility.max()
        min_date, min_value = rolling_avg_volatility.idxmin(), rolling_avg_volatility.min()
        ax.plot(max_date, max_value, 'go')  # Green dot for max
        ax.plot(min_date, min_value, 'ro')  # Red dot for min

        ax.set_title("30-Day Rolling Average Volatility Plot")
        ax.set_xlabel("Date")
        ax.set_ylabel("30-Day Rolling Average Volatility")
        ax.legend()
        return fig
    


    def plot_90_day_rolling_sharpe_ratio(self, data, risk_free_rate):

        #sharpe ratio rolling hard coded for 90 

        daily_returns = data.pct_change().dropna()
        rolling_geometric_mean = daily_returns.rolling(window=90).apply(lambda x: np.exp(np.log1p(x).mean()) - 1)
        rolling_std = daily_returns.rolling(window=90).std()
        # daily_risk_free_rate = (1 + risk_free_rate) ** (1 / (252/90)) - 1 #converting annual rate to a 90 day rate 
        rolling_sharpe_ratio = (rolling_geometric_mean) / rolling_std
        rolling_sharpe_ratio_annualised = rolling_sharpe_ratio * np.sqrt(252/90)
        fig, ax = plt.subplots(figsize=(15, 8))  
        ax.plot(rolling_sharpe_ratio_annualised.index, rolling_sharpe_ratio_annualised, label='90-Day Rolling Sharpe Ratio (Annualised)')
        max_date, max_value = rolling_sharpe_ratio_annualised.idxmax(), rolling_sharpe_ratio_annualised.max()
        min_date, min_value = rolling_sharpe_ratio_annualised.idxmin(), rolling_sharpe_ratio_annualised.min()
        ax.plot(max_date, max_value, 'go')  
        ax.plot(min_date, min_value, 'ro')  
        ax.set_title("90-Day Rolling Sharpe Ratio (Annualised) Plot")
        ax.set_xlabel("Date")
        ax.set_ylabel("90-Day Rolling Sharpe Ratio (Annualised)")
        ax.legend()

        return fig


    def plot_daily_pct_change_with_highlight(self, data, upper_threshold, lower_threshold):
        # Calculate the daily percentage change
        daily_pct_change = data.pct_change().dropna() * 100  # Convert to percentage

        # Create boolean masks for values above the upper threshold and below the lower threshold
        above_threshold = daily_pct_change > upper_threshold
        below_threshold = daily_pct_change < lower_threshold

        # Plot the daily percentage change as a bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(daily_pct_change.index, daily_pct_change, color='blue', edgecolor='black') #allows for better visability when longer time horizon is selected 
        # Highlight the bars that are above the upper threshold or below the lower threshold
        for bar, above, below in zip(bars, above_threshold, below_threshold):
            if above:
                bar.set_color('green')
            elif below:
                bar.set_color('red')
        ax.set_title(f'Daily Percentage Change with Highlight (Upper Threshold: {upper_threshold}%, Lower Threshold: {lower_threshold}%)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Percentage Change')
        ax.grid(True)
        return fig

    def calculate_statistics(self, target, data, risk_free_rate):

        daily_returns = data.pct_change().dropna()
        mean_return = daily_returns.mean()
        num_days_up = (daily_returns > 0).sum()
        num_days_down = (daily_returns < 0).sum()
        percent_days_up = num_days_up / len(daily_returns) * 100
        avg_return_up = daily_returns[daily_returns > 0].mean()
        avg_return_down = daily_returns[daily_returns < 0].mean()
        skewness = daily_returns.skew()
        kurtosis = daily_returns.kurtosis()
        std_dev = daily_returns.std()

        # Calculate the Sharpe ratio Sharpe ratio here is calculated on a daily basis and then annualised 
        daily_risk_free_rate = (1 + risk_free_rate) ** (1 / 252) - 1
        sharpe_ratio = (mean_return - daily_risk_free_rate) / std_dev * np.sqrt(252)

        # Create a DataFrame with the statistics
        stats_df = pd.DataFrame({
            f"{target}": [
            mean_return,
            num_days_up,
            percent_days_up,
            avg_return_up,
            avg_return_down,
            skewness,
            kurtosis,
            std_dev,
            sharpe_ratio
            ]
        }, index=[
            "mean_return",
            "num_days_up",
            "percent_days_up",
            "avg_return_up",
            "avg_return_down",
            "skewness",
            "kurtosis",
            "std_dev",
            "sharpe_ratio"
        ])

        return stats_df

class Rates:
    
    def data(self,selected_horizon):
        data = download_data(["^IRX", "^FVX", "^TNX", "^TYX","TSLA", "^SPX", "GC=F"], selected_horizon)
        data.columns = ["2y", "5y", "10y", "30y","Tesla", "SPX", "Gold"]
        data['2s10s'] = data["10y"]-data["2y"]
        data['5s30s'] = data["30y"]-data["5y"]
        return data

    def plot_rolling_correlation(self, series1, series2, window):
        rolling_corr = series1.rolling(window=window).corr(series2)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(rolling_corr.index, rolling_corr, label=f'Rolling Correlation (Window={window})', color='blue')
        max_date, max_value = rolling_corr.idxmax(), rolling_corr.max()
        min_date, min_value = rolling_corr.idxmin(), rolling_corr.min()
        ax.plot(max_date, max_value, 'go')  
        ax.plot(min_date, min_value, 'ro')  
        ax.set_title('Rolling Correlation Coefficient')
        ax.set_xlabel('Date')
        ax.set_ylabel('Correlation Coefficient')
        ax.legend()
        ax.grid(True)
        return fig

    def calculate_market_value(self, time_series_data, investments):
        market_values = pd.DataFrame(index=time_series_data.index) #make sure the 2 df will have same index 
        time_series_data = time_series_data.dropna()
        for security, investment in investments.items():
            if security in time_series_data.columns:
                # Calculate the number of shares purchased for each security
                shares = investment / time_series_data[security].iloc[0]
                # Calculate the market value over time
                market_values[security] = shares * time_series_data[security]
            else:
                market_values[security] = np.nan
        market_values = market_values.dropna()
        market_values["sum"] = market_values.sum(axis = 1)
        return market_values
    

    def plot_correlation_matrix(self, data):

        data = data.dropna()
        correlation_matrix = data.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(correlation_matrix, cmap='coolwarm')
        fig.colorbar(cax)

        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=90)
        ax.set_yticklabels(correlation_matrix.columns)

        # Annotate the correlation coefficients on the matrix
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                ax.text(j, i, f"{round(correlation_matrix.iloc[i, j],2)}", ha='center', va='center', color='black')

        plt.title('Correlation Matrix', pad=20)
        plt.tight_layout()

        return fig

class Portfolio:
    
    def calculate_market_value(self, time_series_data, investments):

        market_values = pd.DataFrame(index=time_series_data.index)

        time_series_data = time_series_data.dropna()
        for security, investment in investments.items():
            if security in time_series_data.columns:
                shares = investment / time_series_data[security].iloc[0]
                market_values[security] = shares * time_series_data[security]
            else:
                market_values[security] = np.nan
        market_values = market_values.dropna()
        return market_values
    
    def plot_cumulative_sum(self, data1, data2):
        data1 = data1.dropna()
        data2 = data2.dropna()
        cumulative_sum1 = data1.sum(axis=1).to_frame(name="Unhedged")
        cumulative_sum2 = data2.sum(axis=1).to_frame(name="Hedged")

        fig, ax = plt.subplots()

        ax.plot(cumulative_sum1.index, cumulative_sum1['Unhedged'], label='Unhedged')
        ax.plot(cumulative_sum2.index, cumulative_sum2['Hedged'], label='Hedged')

      
        max_date1, max_value1 = cumulative_sum1['Unhedged'].idxmax(), cumulative_sum1['Unhedged'].max()
        min_date1, min_value1 = cumulative_sum1['Unhedged'].idxmin(), cumulative_sum1['Unhedged'].min()
        ax.plot(max_date1, max_value1, 'go')  
        ax.plot(min_date1, min_value1, 'ro')  

        max_date2, max_value2 = cumulative_sum2['Hedged'].idxmax(), cumulative_sum2['Hedged'].max()
        min_date2, min_value2 = cumulative_sum2['Hedged'].idxmin(), cumulative_sum2['Hedged'].min()
        ax.plot(max_date2, max_value2, 'go')  
        ax.plot(min_date2, min_value2, 'ro')  
        ax.set_title("Hedged portfolio vs Unhedged performance")
        ax.set_xlabel("Date")
        ax.set_ylabel("Market value")
        ax.legend()
        return fig

    def plot_efficient_frontier(self, time_series_data, current_weights, calculated, risk_free_rate):

        returns = time_series_data.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        #-----------data prep
        num_portfolios = 10000 # Define the number of portfolios to simulate        
        results = np.zeros((3, num_portfolios)) #init array 
        for i in range(num_portfolios):
            # Generate random portfolio weights
            weights = np.random.random(len(mean_returns))
            weights /= np.sum(weights)
            portfolio_return = np.sum(weights * mean_returns) * 252
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
            results[0, i] = portfolio_return
            results[1, i] = portfolio_volatility
            results[2, i] = (results[0, i] - risk_free_rate)/ results[1, i]  # Sharpe ratio
        max_sharpe_idx = np.argmax(results[2])# Find the portfolio with the maximum Sharpe ratio
        max_sharpe_return = results[0, max_sharpe_idx]
        max_sharpe_volatility = results[1, max_sharpe_idx]
        mean_returns = mean_returns.to_numpy()
        current_return = np.sum(current_weights * mean_returns) * 252
        current_volatility = np.sqrt(np.dot(current_weights.T, np.dot(cov_matrix, current_weights))) * np.sqrt(252)
        # Calculate the current portfolio return and volatility, this is passed in and using minimiser to solve
        calculated_return = np.sum(calculated * mean_returns) * 252
        calculated_volatility = np.sqrt(np.dot(calculated.T, np.dot(cov_matrix, calculated))) * np.sqrt(252)
        # Plot-----------------
        plt.figure(figsize=(10, 6))
        plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', marker='o')
        plt.colorbar(label='Sharpe Ratio')
        plt.scatter(max_sharpe_volatility, max_sharpe_return, marker='*', color='r', s=200, label='Monte carlo max Sharpe Ratio')
        plt.scatter(current_volatility, current_return, marker='o', color='b', s=200, label='Current Portfolio')
        plt.scatter(calculated_volatility, calculated_return, marker='o', color='g', s=200, label='Calculated Portfolio')
        plt.title('Efficient Frontier')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True)
        return plt

    def mean_variance_optimised_portfolio(self, time_series_data, risk_free_rate):
        returns = time_series_data.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        num_assets = len(mean_returns)
        # Define the objective function (negative Sharpe ratio), by fining the minimum negative sharpe, we find the max sharpe 
        def objective(weights):
            portfolio_return = np.sum(weights * mean_returns) * 252
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
            return - (portfolio_return - risk_free_rate) / portfolio_volatility
        # Constraints: weights sum to 1, capital constraint 
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) #specifies that this is a equation
        bounds = tuple((0, 1) for _ in range(num_assets))#does not allow short position 
        initial_guess = num_assets * [1. / num_assets] #Equal weight initial guess
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        optimised_weights = {col: weight for col, weight in zip(time_series_data.columns, result.x)} #2 for loop looping the same time 
        return optimised_weights

    def mean_variance_optimised_portfolio_with_constraints(self, time_series_data, current_weights, risk_free_rate):
        returns = time_series_data.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        num_assets = len(mean_returns)
        def objective(weights):
            portfolio_return = np.sum(weights * mean_returns) * 252
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
            return - (portfolio_return - risk_free_rate) / portfolio_volatility
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # Bounds: allow hedging instrument to take weights smaller than 1
        bounds = []
        for ticker in time_series_data.columns:
            if ticker in ["TSLA", "^SPX", "GC=F"]:
                bounds.append((0, 1))
            else:
                bounds.append((-1, 1))
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        initial_guess = [current_weights.get(ticker, 1. / num_assets) for ticker in time_series_data.columns]
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        optimised_weights = {col: weight for col, weight in zip(time_series_data.columns, result.x)}
        return optimised_weights
    
    def calculate_portfolio_metrics(self, time_series_data, risk_free_rate):
        daily_returns = time_series_data.pct_change().fillna(0) #cannot drop NA!!! will cause empty dataframe
        portfolio_daily_returns = daily_returns.sum(axis=1)
        # Calculating-----------------------all in annualised basis
        mean_return = portfolio_daily_returns.mean() * 252
        volatility = portfolio_daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (mean_return - risk_free_rate) / volatility
        #calmar------
        cumulative_returns = (1 + portfolio_daily_returns).cumprod() #cumulative product will return the cumulative return in percentage 
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        calmar_ratio = mean_return / abs(max_drawdown)

        gain_given_positive = portfolio_daily_returns[portfolio_daily_returns > 0].mean()
        loss_given_negative = portfolio_daily_returns[portfolio_daily_returns < 0].mean()

        kurtosis = portfolio_daily_returns.kurtosis()
        skew = portfolio_daily_returns.skew()

        percent_positive_days = (portfolio_daily_returns > 0).mean() * 100
        var_95 = np.percentile(portfolio_daily_returns, 5)

        return {
            "mean_return": round(mean_return, 2),
            "volatility": round(volatility, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "calmar_ratio": round(calmar_ratio, 2),
            "gain_given_positive": round(gain_given_positive, 2),
            "loss_given_negative": round(loss_given_negative, 2),
            "kurtosis": round(kurtosis, 2),
            "skew": round(skew, 2),
            "percent_positive_days": round(percent_positive_days, 2),
            "var_95": round(var_95, 2)
        }
    
    def plot_histogram_with_normal_fit(self, df1, df2):
        daily_changes1 = df1.sum(axis=1).pct_change().dropna()
        daily_changes2 = df2.sum(axis=1).pct_change().dropna()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(daily_changes1, bins=50, alpha=0.5, label='Unhedged', density=True, color='blue', edgecolor='black')
        ax.hist(daily_changes2, bins=50, alpha=0.5, label='Hedged', density=True, color='red', edgecolor='black')
        #Add normal bell curve-------------------
        mu1, std1 = norm.fit(daily_changes1)
        x1 = np.linspace(-3*std1, 3*std1, 100) + mu1
        p1 = norm.pdf(x1, mu1, std1)
        ax.plot(x1, p1, 'b--', linewidth=2)
        mu2, std2 = norm.fit(daily_changes2)
        x2 = np.linspace(-3*std2, 3*std2, 100) + mu2
        p2 = norm.pdf(x2, mu2, std2)
        ax.plot(x2, p2, 'r--', linewidth=2)
        ax.set_title('Histogram of Daily price Changes')
        ax.set_xlabel('Daily Change')
        ax.set_ylabel('Density')
        ax.legend()
        return fig

    def plot_comparison_bar_chart_position(self, dict1, dict2):
        df = pd.DataFrame({'Optimised portfolio': dict1, 'Current portfolio': dict2})
        df = df.apply(pd.to_numeric, errors='coerce')
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(kind='bar', ax=ax)
        ax.set_title('Portfolio allocations')
        ax.set_xlabel('Categories')
        ax.set_ylabel('ratios')
        ax.legend(title='Portfolios')
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() * 1.005, p.get_height() * 1.005))
        plt.tight_layout()
        return fig
    
    def plot_comparison_bar_chart_metrics(self, dict1, dict2):
        df = pd.DataFrame({'Hedged portfolio': dict1, 'Current portfolio': dict2})
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.drop(['percent_positive_days', 'kurtosis'])#dropping because of scaling probvlems 
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(kind='bar', ax=ax)
        ax.set_title('portfolio performances')
        ax.set_xlabel('Categories')
        ax.set_ylabel('In there respective units')
        ax.legend(title='Portfolios')
        plt.tight_layout()
        return fig

     