import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import os

os.makedirs('outputs/charts', exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f8f9fa',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
})

FEAR_COLOR    = '#E63946'
GREED_COLOR   = '#2DC653'
NEUTRAL_COLOR = '#457B9D'
PALETTE       = {'Fear': FEAR_COLOR, 'Greed': GREED_COLOR}

trades    = pd.read_csv('data/trader_data.csv')
sentiment = pd.read_csv('data/sentiment_data.csv')

trades['time']    = pd.to_datetime(trades['time'])
sentiment['Date'] = pd.to_datetime(sentiment['Date'])
trades['date']    = trades['time'].dt.date
sentiment['date'] = sentiment['Date'].dt.date

df = trades.merge(sentiment[['date', 'Classification']], on='date', how='left')
df['Classification'] = df['Classification'].ffill()

df['is_profit']     = df['closedPnL'] > 0
df['size_category'] = pd.qcut(df['size'], q=4, labels=['small','medium','large','whale'])
df['high_leverage'] = df['leverage'] > df['leverage'].median()
df['hour']          = df['time'].dt.hour
df['day_of_week']   = df['time'].dt.day_name()
df['month']         = df['time'].dt.to_period('M')

daily_vol           = df.groupby('date')['closedPnL'].std().reset_index()
daily_vol.columns   = ['date', 'daily_volatility']
df                  = df.merge(daily_vol, on='date', how='left')

print("Shape:", df.shape)
print("Sentiment split:\n", df['Classification'].value_counts())

profit_summary = df.groupby('Classification')['closedPnL'].agg(
    avg_pnl='mean', total_pnl='sum', median_pnl='median',
    std_pnl='std', trade_count='count'
).round(2)
print("\nPnL Summary:\n", profit_summary)

win_rate = (df.groupby('Classification')['is_profit'].mean() * 100).round(2)
print("\nWin Rate:\n", win_rate)

lev_analysis = df.groupby('Classification')['leverage'].agg(['mean','median','std']).round(2)
print("\nLeverage:\n", lev_analysis)

direction = (pd.crosstab(df['Classification'], df['side'], normalize='index') * 100).round(1)
print("\nDirection Bias:\n", direction)

df_clip = df.copy()
df_clip['closedPnL'] = df_clip['closedPnL'].clip(-150, 150)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for s, c in [('Fear', FEAR_COLOR), ('Greed', GREED_COLOR)]:
    axes[0].hist(df_clip[df_clip['Classification']==s]['closedPnL'],
                 bins=60, alpha=0.65, color=c, density=True, label=s)
axes[0].set_title('PnL Distribution by Sentiment', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Closed PnL (USD)')
axes[0].axvline(0, color='gray', linestyle='--', alpha=0.6)
axes[0].legend()
sns.boxplot(x='Classification', y='closedPnL', data=df_clip,
            palette=PALETTE, ax=axes[1],
            medianprops=dict(color='white', linewidth=2.5))
axes[1].set_title('PnL Boxplot by Sentiment', fontsize=14, fontweight='bold')
axes[1].axhline(0, color='gray', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('outputs/charts/01_pnl_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(win_rate.index, win_rate.values,
              color=[FEAR_COLOR, GREED_COLOR], width=0.45, edgecolor='white')
for bar, val in zip(bars, win_rate.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
            f'{val:.1f}%', ha='center', fontsize=13, fontweight='bold')
ax.axhline(50, color='gray', linestyle='--', label='50% baseline')
ax.set_ylim(0, 70)
ax.set_ylabel('Win Rate (%)')
ax.set_title('Win Rate by Market Sentiment', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/charts/02_win_rate.png', dpi=150, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
for s, c in [('Fear', FEAR_COLOR), ('Greed', GREED_COLOR)]:
    sub = df[df['Classification']==s]['leverage']
    ax.hist(sub, bins=50, alpha=0.65, color=c, density=True,
            label=f'{s} (mean={sub.mean():.1f}x)')
    ax.axvline(sub.mean(), color=c, linestyle='--', linewidth=2.2)
ax.set_xlabel('Leverage (x)')
ax.set_title('Leverage Distribution by Sentiment', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/charts/03_leverage.png', dpi=150, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for i, (s, c) in enumerate([('Fear', FEAR_COLOR), ('Greed', GREED_COLOR)]):
    sub = df[df['Classification']==s]['side'].value_counts()
    axes[i].pie(sub.values, labels=sub.index, autopct='%1.1f%%',
                colors=[c, '#F4A261'],
                wedgeprops=dict(edgecolor='white', linewidth=2),
                textprops={'fontsize': 12})
    axes[i].set_title(f'{s} Market', fontsize=13, fontweight='bold', color=c)
plt.suptitle('Buy vs Sell by Sentiment', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/charts/04_direction.png', dpi=150, bbox_inches='tight')
plt.show()

monthly = df.groupby(['month','Classification'])['closedPnL'].mean().reset_index()
monthly['month_str'] = monthly['month'].astype(str)
fig, ax = plt.subplots(figsize=(13, 5))
for s, c in [('Fear', FEAR_COLOR), ('Greed', GREED_COLOR)]:
    sub = monthly[monthly['Classification']==s]
    ax.plot(sub['month_str'], sub['closedPnL'],
            marker='o', color=c, linewidth=2.5, markersize=6, label=s)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Monthly Avg PnL Trend by Sentiment', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Avg PnL (USD)')
ax.legend()
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig('outputs/charts/05_monthly_trend.png', dpi=150, bbox_inches='tight')
plt.show()

pivot = df.groupby(['symbol','Classification'])['closedPnL'].mean().unstack()
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
            linewidths=0.5, linecolor='white', ax=ax,
            cbar_kws={'label': 'Avg PnL (USD)'})
ax.set_title('Avg PnL by Symbol & Sentiment', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/charts/06_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

corr_cols = ['closedPnL','leverage','size','daily_volatility']
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.3f',
            cmap='coolwarm', center=0, linewidths=0.5, ax=ax)
ax.set_title('Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/charts/07_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

trader_perf = df.groupby('account').agg(
    total_pnl    = ('closedPnL','sum'),
    win_rate     = ('is_profit','mean'),
    avg_leverage = ('leverage','mean'),
    trade_count  = ('closedPnL','count'),
).reset_index()

features = ['total_pnl','win_rate','avg_leverage']
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(trader_perf[features])
kmeans   = KMeans(n_clusters=3, random_state=42, n_init=10)
trader_perf['cluster'] = kmeans.fit_predict(X_scaled)

means     = trader_perf.groupby('cluster')['total_pnl'].mean().sort_values(ascending=False)
label_map = {means.index[0]:'Smart Money', means.index[1]:'Neutral Traders', means.index[2]:'High-Risk Losers'}
trader_perf['cluster_label'] = trader_perf['cluster'].map(label_map)

cluster_colors = {'Smart Money':GREED_COLOR,'Neutral Traders':NEUTRAL_COLOR,'High-Risk Losers':FEAR_COLOR}
fig, ax = plt.subplots(figsize=(10, 6))
for label, grp in trader_perf.groupby('cluster_label'):
    ax.scatter(grp['avg_leverage'], grp['total_pnl'],
               color=cluster_colors[label], label=label, alpha=0.75, s=70, edgecolors='white')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Average Leverage (x)')
ax.set_ylabel('Total PnL (USD)')
ax.set_title('Trader Clustering: 3 Archetypes', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/charts/08_clustering.png', dpi=150, bbox_inches='tight')
plt.show()

trader_perf['score'] = (
    trader_perf['total_pnl']    * 0.5 +
    trader_perf['win_rate']     * 1000 * 0.3 -
    trader_perf['avg_leverage'] * 10   * 0.2
)
top10 = trader_perf.sort_values('score', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(range(10), top10['total_pnl'].values,
               color=GREED_COLOR, edgecolor='white')
ax.set_yticks(range(10))
ax.set_yticklabels([f'Trader {i+1}' for i in range(10)])
ax.invert_yaxis()
ax.set_xlabel('Total PnL (USD)')
ax.set_title('Top 10 Traders by Performance Score', fontsize=14, fontweight='bold')
for bar, val in zip(bars, top10['total_pnl'].values):
    ax.text(bar.get_width()+5, bar.get_y()+bar.get_height()/2,
            f'${val:,.0f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('outputs/charts/09_top_traders.png', dpi=150, bbox_inches='tight')
plt.show()

def strategy_recommendation(row):
    if row['Classification'] == 'Greed' and row['leverage'] > 10:
        return 'High Risk — Reduce Leverage'
    elif row['Classification'] == 'Fear' and row['side'] == 'Buy':
        return 'Optimal Dip Buy'
    elif row['Classification'] == 'Fear' and row['side'] == 'Sell':
        return 'Caution — Panic Sell'
    else:
        return 'Neutral'

df['strategy'] = df.apply(strategy_recommendation, axis=1)

strat_perf = df.groupby('strategy').agg(
    count   = ('closedPnL','count'),
    avg_pnl = ('closedPnL','mean'),
    win_rate= ('is_profit','mean')
).round(2).sort_values('avg_pnl', ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
colors = [GREED_COLOR if v > 0 else FEAR_COLOR for v in strat_perf['avg_pnl']]
bars = ax.barh(strat_perf.index, strat_perf['avg_pnl'], color=colors, edgecolor='white')
ax.axvline(0, color='gray', linestyle='--', alpha=0.6)
ax.set_xlabel('Avg PnL per Trade (USD)')
ax.set_title('Avg PnL by Strategy Tag', fontsize=14, fontweight='bold')
for bar, val in zip(bars, strat_perf['avg_pnl']):
    ax.text(val+(0.05 if val>=0 else -0.05),
            bar.get_y()+bar.get_height()/2,
            f'${val:.2f}', va='center',
            ha='left' if val>=0 else 'right', fontsize=9)
plt.tight_layout()
plt.savefig('outputs/charts/10_strategy.png', dpi=150, bbox_inches='tight')
plt.show()

profit_summary.to_csv('outputs/summary_pnl.csv')
trader_perf.groupby('cluster_label').agg(
    count=('account','count'),
    avg_pnl=('total_pnl','mean'),
    avg_win_rate=('win_rate','mean'),
    avg_leverage=('avg_leverage','mean')
).round(2).to_csv('outputs/summary_clusters.csv')
strat_perf.to_csv('outputs/summary_strategies.csv')
top10.to_csv('outputs/top10_traders.csv', index=False)

print("\nDone. Charts saved to outputs/charts/")
