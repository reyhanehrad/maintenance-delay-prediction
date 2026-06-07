"""
==========================================================================
SCRIPT 02: Exploratory Data Analysis
==========================================================================
Following the professor's EDA approach (PDF pages 9-10):
- visualize key relationships
- compute correlation matrix
- identify patterns for ANN modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style('whitegrid')

df = pd.read_csv('milano_cleaned.csv', low_memory=False)
print(f"Loaded {len(df):,} cleaned records")

# ==========================================================================
# FIGURE 1: Dataset Overview Dashboard
# ==========================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('POLIMI Milano Maintenance Tickets — Dataset Overview',
             fontsize=14, fontweight='bold')

# (1,1) Class balance
ax = axes[0, 0]
counts = df['is_delayed'].value_counts().sort_index()
colors = ['#4CAF50', '#F44336']
bars = ax.bar(['On Time', 'Delayed'], counts.values, color=colors, edgecolor='black')
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', fontweight='bold')
ax.set_title('Ticket Outcome Distribution')
ax.set_ylabel('Number of Tickets')
ax.set_ylim(0, max(counts.values) * 1.18)

# (1,2) Urgency distribution
ax = axes[0, 1]
urg_counts = df['Urgency'].value_counts()
ax.bar(urg_counts.index, urg_counts.values,
       color=['#2196F3', '#FF9800', '#F44336'], edgecolor='black')
ax.set_title('Urgency Level Distribution')
ax.set_ylabel('Number of Tickets')
for i, v in enumerate(urg_counts.values):
    ax.text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')

# (1,3) SLA distribution (top 8)
ax = axes[0, 2]
sla_counts = df['Expiration Time'].value_counts().head(8)
ax.barh(sla_counts.index[::-1], sla_counts.values[::-1],
        color='#3F51B5', edgecolor='black')
ax.set_title('SLA (Expiration Time) — Top 8')
ax.set_xlabel('Number of Tickets')

# (2,1) Top 10 macro categories
ax = axes[1, 0]
cat_counts = df['macro_category'].value_counts().head(10)
ax.barh(cat_counts.index[::-1], cat_counts.values[::-1],
        color='#009688', edgecolor='black')
ax.set_title('Top 10 Maintenance Macro-Categories')
ax.set_xlabel('Number of Tickets')

# (2,2) Top 10 buildings
ax = axes[1, 1]
b_counts = df['ID_Building'].value_counts().head(10)
ax.bar(range(len(b_counts)), b_counts.values, color='#9C27B0', edgecolor='black')
ax.set_xticks(range(len(b_counts)))
ax.set_xticklabels(b_counts.index, rotation=45)
ax.set_title('Top 10 Buildings by Ticket Volume')
ax.set_ylabel('Number of Tickets')
ax.set_xlabel('Building ID')

# (2,3) Tickets per month
ax = axes[1, 2]
month_counts = df['opening_month'].value_counts().sort_index()
ax.plot(month_counts.index, month_counts.values, marker='o',
        linewidth=2, markersize=8, color='#FF5722')
ax.fill_between(month_counts.index, month_counts.values, alpha=0.3, color='#FF5722')
ax.set_title('Tickets Opened per Month (2023)')
ax.set_xlabel('Month')
ax.set_ylabel('Number of Tickets')
ax.set_xticks(range(1, 13))

plt.tight_layout()
plt.savefig('fig1_overview.png', bbox_inches='tight')
print("✓ fig1_overview.png saved")
plt.close()

# ==========================================================================
# FIGURE 2: Delay distribution & relationships
# ==========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Delay Patterns Analysis', fontsize=14, fontweight='bold')

# (1,1) Delay distribution (capped for visibility)
ax = axes[0, 0]
delays = df['delay_hours'].clip(-100, 1000)
ax.hist(delays, bins=80, color='#3F51B5', edgecolor='black', alpha=0.85)
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='SLA Boundary')
ax.set_xlabel('Delay (hours) — clipped at 1000h for visibility')
ax.set_ylabel('Frequency')
ax.set_title(f'Delay Distribution (median = {df["delay_hours"].median():.0f}h)')
ax.legend()

# (1,2) Delay rate by Urgency
ax = axes[0, 1]
urg_delay = df.groupby('Urgency')['is_delayed'].mean().sort_values(ascending=False) * 100
bars = ax.bar(urg_delay.index, urg_delay.values,
              color=['#F44336', '#FF9800', '#2196F3'], edgecolor='black')
for bar, val in zip(bars, urg_delay.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', fontweight='bold')
ax.set_title('Delay Rate by Urgency')
ax.set_ylabel('% of Tickets Delayed')
ax.set_ylim(0, 100)

# (2,1) Delay rate by SLA
ax = axes[1, 0]
sla_delay = df.groupby('SLA_hours')['is_delayed'].mean().sort_index() * 100
sla_count = df.groupby('SLA_hours').size().sort_index()
ax.bar(range(len(sla_delay)), sla_delay.values, color='#009688', edgecolor='black')
labels = [f'{h:.2f}h\n(n={c:,})' if h < 1 else f'{int(h)}h\n(n={c:,})'
          for h, c in zip(sla_delay.index, sla_count.values)]
ax.set_xticks(range(len(sla_delay)))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_title('Delay Rate by SLA Length')
ax.set_ylabel('% of Tickets Delayed')
ax.set_ylim(0, 100)

# (2,2) Delay rate by macro-category (top 10 by volume)
ax = axes[1, 1]
top_cats = df['macro_category'].value_counts().head(10).index
cat_delay = df[df['macro_category'].isin(top_cats)].groupby('macro_category')['is_delayed'].mean() * 100
cat_delay = cat_delay.sort_values(ascending=True)
ax.barh(cat_delay.index, cat_delay.values, color='#9C27B0', edgecolor='black')
ax.set_title('Delay Rate by Macro-Category (top 10)')
ax.set_xlabel('% of Tickets Delayed')
ax.set_xlim(0, 100)
for i, (cat, val) in enumerate(cat_delay.items()):
    ax.text(val + 1, i, f'{val:.1f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('fig2_delay_patterns.png', bbox_inches='tight')
print("✓ fig2_delay_patterns.png saved")
plt.close()

# ==========================================================================
# FIGURE 3: Correlation Matrix (numerical features) - mimics PDF page 10
# ==========================================================================
numeric_cols = ['SLA_hours', 'urgency_level', 'opening_month',
                'opening_dayofweek', 'opening_hour',
                'ID_Building', 'ID_BuildingGroup',
                'delay_hours', 'is_delayed']
corr = df[numeric_cols].corr().round(2)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('Correlation Matrix — Numerical Features',
             fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('fig3_correlation.png', bbox_inches='tight')
print("✓ fig3_correlation.png saved")
plt.close()

# ==========================================================================
# Print key findings
# ==========================================================================
print("\n" + "=" * 70)
print("KEY EDA FINDINGS")
print("=" * 70)

print(f"\n[1] Overall delay rate: {df['is_delayed'].mean()*100:.1f}%")
print(f"    Median delay : {df['delay_hours'].median():.0f}h")
print(f"    Mean delay   : {df['delay_hours'].mean():.0f}h")

print(f"\n[2] Delay rate by urgency:")
for urg, rate in df.groupby('Urgency')['is_delayed'].mean().items():
    print(f"    {urg:15s}: {rate*100:.1f}%")

print(f"\n[3] Top correlations with is_delayed:")
top_corr = corr['is_delayed'].drop('is_delayed').abs().sort_values(ascending=False).head(5)
for var, c in top_corr.items():
    sign = '+' if corr.loc[var, 'is_delayed'] > 0 else '-'
    print(f"    {var:25s}: {sign}{c:.2f}")

print(f"\n[4] Buildings: {df['ID_Building'].nunique()} unique")
print(f"    Most-ticketed building: {df['ID_Building'].value_counts().idxmax()} "
      f"({df['ID_Building'].value_counts().max():,} tickets)")
