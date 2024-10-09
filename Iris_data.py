import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = sns.load_dataset('iris')

# Display basic information
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Univariate Analysis - Distribution of Features
df.hist(figsize=(10, 8))
plt.suptitle('Distribution of Features', fontsize=16)
plt.show()

# Box plot to show data distribution and outliers
sns.boxplot(data=df, orient='h')
plt.title("Box plot of Iris dataset features")
plt.show()

# Pairplot to show relationships between variables
sns.pairplot(df, hue="species")
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()

# Heatmap to show correlation between features
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=2)
plt.title("Heatmap of Iris dataset")
plt.show()
