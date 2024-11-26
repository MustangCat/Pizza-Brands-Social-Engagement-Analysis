# 🍕 pizza-brands-data-analysis

[See the full report](https://pizza-brands-data-analysis.my.canva.site/)

---

## 1. Situation

**Pizza Brands Engagement Analysis**  

Who doesn’t love pizza? When people on social media mention popular pizza brands in Thailand—**Pizza Company**, **Pizza Hut**, and **Domino’s**—it ignites cravings that often lead to a quick pizza order. Engagement plays a crucial role for these brands. They know the importance of creating buzz and fostering interaction to build a strong connection with their audience. But how can these brands improve their engagement even further?

---

## 2. Metadata Overview

Below is the structure of the dataset used for this analysis:

| #   | Column           | Non-Null Count  | Dtype  |
|-----|------------------|-----------------|--------|
| 0   | Account          | 2273 non-null  | object |
| 1   | Message          | 2274 non-null  | object |
| 2   | Source           | 2274 non-null  | object |
| 3   | post date        | 2274 non-null  | object |
| 4   | post time        | 2274 non-null  | object |
| 5   | time cate.       | 2274 non-null  | object |
| 6   | post day         | 2274 non-null  | object |
| 7   | Engagement       | 2274 non-null  | int64  |
| 8   | Main keyword     | 2274 non-null  | object |
| 9   | Follower count   | 2274 non-null  | object |
| 10  | Sentiment        | 2274 non-null  | object |
| 11  | Category         | 2274 non-null  | object |
| 12  | _id              | 2274 non-null  | object |
| 13  | Image labels     | 2274 non-null  | object |
| 14  | Logo detections  | 2274 non-null  | object |

---

## 3. Data Manipulation Steps

- **Handling Missing Data**
  - Filled in missing values in the **Category** column by extracting relevant text from the **Main Keyword** field.

- **Data Type Conversion**
  - Converted the `Follower count` column from `object` to `int`.

- **Time-Based Features**
  - Extracted **date**, **time**, and **day of the week** from the `post time` column for better temporal analysis.

- **Category Follower Group**
  - Small: For accounts with **fewer than 10,000** followers.
    These are typically individuals or niche accounts with a smaller audience.

  - Medium: For accounts with **10,000 to 100,000** followers.
    These are mid-tier accounts, often considered micro-influencers or medium-scale brands.

  - Large: For accounts with **more than 100,000** followers.
    These are prominent influencers, celebrities, or large brands with a substantial audience.

---

## 4. Data Analysis Challenges and Solutions

### 4.1. Imbalanced Categories
- **Observation**  
Messages are distributed unevenly among the brands:  
  - Pizza Company: **1,210** messages  
  - Pizza Hut: **910** messages  
  - Domino’s: **154** messages  

- **Analysis**  
This imbalance is not due to sampling bias, as the time period covered (2 months) was consistent for all brands.

- **Solution**  
Separate analyses were conducted for each brand to ensure insights are meaningful.

---

### 4.2. Outlier Detection and Handling

Outliers were removed using the following logic:

**Data Cleaning**  
Entries with non-positive Follower count or Engagement values were excluded.

**Outlier Thresholds**  
- **Follower Threshold:** The top 5% of follower counts were identified as potential outliers.  
- **Engagement Threshold:** The bottom 25% of engagement rates were considered low engagement.

**Outlier Filtering**  
Entries with high follower counts (above the 95th percentile) and low engagement (below the 25th percentile) were removed, as they indicate an anomaly (e.g., inactive or irrelevant influencers).

```python
# Remove entries with missing or invalid follower or engagement data
pizza_cleaned = pizza[(pizza['Follower count'] > 0) & (pizza['Engagement'] >= 0)]

# Define outlier thresholds
follower_threshold = pizza_cleaned['Follower count'].quantile(0.95)  # Top 5% followers
engagement_threshold = pizza_cleaned['Engagement'].quantile(0.25)    # Bottom 25% engagement

# Filter out high-follower, low-engagement outliers
filtered_data = pizza_cleaned[
    ~((pizza_cleaned['Follower count'] > follower_threshold) & 
      (pizza_cleaned['Engagement'] < engagement_threshold))
]

---

## 5. Key Findings

### **Engagement Insights**
- **Pizza Company** had the highest average engagement rate per post, significantly outperforming **Pizza Hut** and **Domino’s**. 
- Posts mentioning **Pizza Hut** had higher engagement during weekends, suggesting a stronger audience connection on leisure days.
- **Domino's** posts showed an interesting trend: although they had the fewest messages, they achieved a high engagement-to-message ratio, highlighting a more niche but active audience.

### **Content-Type Patterns**
- Posts with images, especially featuring the brand logo or food visuals, drove **40% higher engagement** than those without images.
- Posts with positive sentiment yielded **3x higher engagement rates** compared to neutral or negative posts.

### **Time-Based Trends**
- Posts between **6 PM and 9 PM** consistently showed higher engagement rates, aligning with dinner-time browsing habits.

---

## 6. Recommendations

### **Content Strategy**
- Leverage food visuals and branding prominently in posts to maximize engagement.
- Experiment with sentiment-driven content (e.g., emphasizing joy or satisfaction) to capitalize on positive emotional triggers.

### **Timing Optimization**
- Schedule posts during peak engagement hours (6 PM–9 PM) and focus on weekends for **Pizza Hut** to boost visibility.
- Analyze engagement data monthly to refine timing strategies dynamically.

### **Target Audience Focus**
- For **Pizza Company**: Invest in campaigns that cater to their broad audience to maintain and expand their engagement lead.
- For **Domino’s**: Focus on strengthening their niche, possibly through targeted social media ads and exclusive offers for their loyal followers.
- For **Pizza Hut**: Highlight promotions and themed campaigns during weekends.

---
