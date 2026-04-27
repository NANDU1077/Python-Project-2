# 📦 Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt

# 🔄 Load the Dataset
df = pd.read_csv("/Users/nandu/Downloads/Shill Bidding Dataset.csv")

# 🎯 Objective 1: Count of Legitimate vs Shill Bidding (Bar Graph)
plt.figure(figsize=(6, 4))
class_counts = df['Class'].value_counts().sort_index()
plt.bar(['Legitimate (0)', 'Shill (1)'], class_counts.values, color=['skyblue', 'salmon'])
plt.title('Count of Legitimate vs Shill Bidding')
plt.xlabel('Class')
plt.ylabel('Number of Records')
plt.grid(True)
plt.tight_layout()
plt.show()

# 🎯 Objective 2: Average Auction Bids by Class (Horizontal Bar Chart)
plt.figure(figsize=(6, 4))
avg_bids = df.groupby('Class')['Auction_Bids'].mean()
plt.barh(['Legitimate (0)', 'Shill (1)'], avg_bids.values, color=['skyblue', 'salmon'])
plt.title('Average Auction Bids by Class')
plt.xlabel('Average Auction Bids')
plt.grid(True)
plt.tight_layout()
plt.show()

# 🎯 Objective 3: Distribution of Early Bidding (Histogram)
plt.figure(figsize=(6, 4))
plt.hist(df['Early_Bidding'], bins=20, color='purple', edgecolor='black')
plt.title('Distribution of Early Bidding')
plt.xlabel('Early Bidding')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# 🎯 Objective 4: Early vs Last Bidding (Scatter Plot)
plt.figure(figsize=(6, 4))
plt.scatter(df['Early_Bidding'], df['Last_Bidding'], alpha=0.5, c=df['Class'], cmap='coolwarm')
plt.title('Early vs Last Bidding (Colored by Class)')
plt.xlabel('Early Bidding')
plt.ylabel('Last Bidding')
plt.colorbar(label='Class (0 = Legit, 1 = Shill)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 🎯 Objective 5: Average Winning Ratio over Auction Duration (Line Graph)
plt.figure(figsize=(6, 4))
avg_win_ratio = df.groupby('Auction_Duration')['Winning_Ratio'].mean()
plt.plot(avg_win_ratio.index, avg_win_ratio.values, marker='o', linestyle='-', color='green')
plt.title('Average Winning Ratio over Auction Duration')
plt.xlabel('Auction Duration (days)')
plt.ylabel('Average Winning Ratio')
plt.grid(True)
plt.tight_layout()
plt.show()

# 🎯 Module 6: Bidding Ratio Distribution by Class (Box Plot)
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.boxplot(x='Class', y='Bidding_Ratio', data=df, palette=['skyblue', 'salmon'])
plt.title('Distribution of Bidding Ratio by Class')
plt.xticks([0, 1], ['Legitimate (0)', 'Shill (1)'])
plt.xlabel('Class')
plt.ylabel('Bidding Ratio')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 🎯 Module 7: Feature Correlation Heatmap
import seaborn as sns

plt.figure(figsize=(12, 8))
# Exclude ID columns as they are not useful for correlation
numeric_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=['Record_ID', 'Auction_ID'])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Bidding Features')
plt.tight_layout()
plt.show()
