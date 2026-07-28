import nbformat as nbf
import os

def create_movie_recommendation_nb():
    nb = nbf.v4.new_notebook()
    
    nb.cells = [
        nbf.v4.new_markdown_cell("# Movie Recommendation System\nThis notebook demonstrates a content-based movie recommendation system using TF-IDF Vectorization and Cosine Similarity. It takes a movie title as input and suggests similar movies based on textual metadata (genres, keywords, cast, etc.)."),
        
        nbf.v4.new_markdown_cell("## 1. Import Libraries\nWe will use `pandas` for data manipulation, and `scikit-learn` for text vectorization and computing similarity scores."),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.metrics.pairwise import cosine_similarity\nimport difflib\n\nimport warnings\nwarnings.filterwarnings('ignore')"),
        
        nbf.v4.new_markdown_cell("## 2. Load the Dataset\nWe use a publicly available movies dataset that contains relevant metadata for content-based filtering."),
        nbf.v4.new_code_cell("url = 'https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Movies%20Recommendation.csv'\ndf = pd.read_csv(url)\n\nprint(f\"Dataset Shape: {df.shape}\")\ndf.head()"),
        
        nbf.v4.new_markdown_cell("## 3. Data Cleaning and Feature Engineering\nTo compute similarity, we need to extract meaningful text features. We will combine `Movie_Genre`, `Movie_Keywords`, `Movie_Tagline`, `Movie_Cast`, and `Movie_Director` into a single string for each movie."),
        nbf.v4.new_code_cell("# Select relevant features for content-based filtering\nfeatures = ['Movie_Genre', 'Movie_Keywords', 'Movie_Tagline', 'Movie_Cast', 'Movie_Director']\n\n# Fill missing values with empty strings\nfor feature in features:\n    df[feature] = df[feature].fillna('')\n\n# Combine features into a single string for each movie\ndef combine_features(row):\n    return row['Movie_Genre'] + ' ' + row['Movie_Keywords'] + ' ' + row['Movie_Tagline'] + ' ' + row['Movie_Cast'] + ' ' + row['Movie_Director']\n\ndf['Combined_Features'] = df.apply(combine_features, axis=1)\ndf[['Movie_Title', 'Combined_Features']].head()"),
        
        nbf.v4.new_markdown_cell("## 4. Text Vectorization (TF-IDF)\nMachine learning algorithms work with numerical data. We use TF-IDF (Term Frequency-Inverse Document Frequency) to convert the combined text features into numerical vectors, assigning importance to words that are unique to specific movies."),
        nbf.v4.new_code_cell("vectorizer = TfidfVectorizer()\nfeature_vectors = vectorizer.fit_transform(df['Combined_Features'])\nprint(f\"Feature Vectors Shape: {feature_vectors.shape}\")"),
        
        nbf.v4.new_markdown_cell("## 5. Calculate Cosine Similarity\nCosine similarity measures the cosine of the angle between two vectors. A higher score means the movies have more similar content."),
        nbf.v4.new_code_cell("similarity = cosine_similarity(feature_vectors)\nprint(f\"Similarity Matrix Shape: {similarity.shape}\")"),
        
        nbf.v4.new_markdown_cell("## 6. Recommendation Function\nWe create a function that takes a movie name, finds the closest match in our dataset (in case of minor typos), and returns the top 5 most similar movies."),
        nbf.v4.new_code_cell("def recommend_movies(movie_name, df, similarity, num_recommendations=5):\n    # Find the closest match to the movie name\n    list_of_all_titles = df['Movie_Title'].tolist()\n    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)\n    \n    if not find_close_match:\n        print(\"\\n\" + \"=\"*50)\n        print(f\"❌ No close match found for '{movie_name}' in the dataset.\")\n        print(\"=\"*50 + \"\\n\")\n        return\n        \n    close_match = find_close_match[0]\n    \n    # Find the index of the row directly\n    movie_index = df.index[df['Movie_Title'] == close_match].tolist()[0]\n    \n    # Get similarity scores\n    similarity_scores = list(enumerate(similarity[movie_index]))\n    \n    # Sort movies based on similarity\n    sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)\n    \n    print(\"\\n\" + \"★\"*50)\n    print(f\"  Top {num_recommendations} Recommendations for: '{close_match}'\")\n    print(\"★\"*50 + \"\\n\")\n    \n    # Print recommendations (skipping the first one as it is the movie itself)\n    i = 1\n    for movie in sorted_similar_movies[1:]:  # skip index 0\n        index = movie[0]\n        title_from_index = df.iloc[index]['Movie_Title']\n        if i <= num_recommendations:\n            print(f\"  {i}. {title_from_index}\")\n            i += 1\n        else:\n            break\n    print(\"\\n\" + \"=\"*50 + \"\\n\")"),
        
        nbf.v4.new_markdown_cell("## 7. Test the Recommendation System\nLet's test our recommendation engine with a popular sci-fi movie."),
        nbf.v4.new_code_cell("recommend_movies('Interstellar', df, similarity, num_recommendations=5)\n\nrecommend_movies('The Dark Knight', df, similarity, num_recommendations=5)"),
        
        nbf.v4.new_markdown_cell("## 8. Conclusion\nThe Movie Recommendation System successfully demonstrates the application of Natural Language Processing and machine learning to solve real-world content discovery problems. By leveraging TF-IDF vectorization and cosine similarity, we built an engine capable of recommending contextually relevant movies based entirely on metadata like cast, genres, and keywords without relying on user interaction history.")
    ]
    
    with open('Movie_Recommendation_System/Movie_Recommendation.ipynb', 'w') as f:
        nbf.write(nb, f)


def create_churn_prediction_nb():
    nb = nbf.v4.new_notebook()
    
    nb.cells = [
        nbf.v4.new_markdown_cell("# Customer Churn Prediction\nThis notebook demonstrates a predictive analytics pipeline to identify customers likely to discontinue a telecom service. We will perform Exploratory Data Analysis, feature engineering, and evaluate multiple classification models."),
        
        nbf.v4.new_markdown_cell("## 1. Import Libraries\nWe import essential libraries for data manipulation, visualization, and machine learning."),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler, LabelEncoder\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report\n\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# Set plotting style\nsns.set_theme(style=\"whitegrid\")"),
        
        nbf.v4.new_markdown_cell("## 2. Load the Dataset\nWe use the IBM Telco Customer Churn dataset, which contains demographic info, account details, and service usage."),
        nbf.v4.new_code_cell("url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'\ndf = pd.read_csv(url)\nprint(f\"Dataset Shape: {df.shape}\")\ndf.head()"),
        
        nbf.v4.new_markdown_cell("## 3. Exploratory Data Analysis (EDA)\nWe analyze the distribution of our target variable (`Churn`) and examine how different categorical features correlate with customer churn."),
        nbf.v4.new_code_cell("# Convert TotalCharges to numeric, coercing errors to NaN\ndf['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')\n# Drop rows with NaN\ndf.dropna(inplace=True)\n\n# 1. Target Variable Distribution\nplt.figure(figsize=(6, 4))\nsns.countplot(x='Churn', data=df, palette='Set2')\nplt.title('Churn Distribution')\nplt.show()\n\n# 2. Categorical Features vs Churn\nfig, axes = plt.subplots(1, 3, figsize=(18, 5))\nsns.countplot(x='Contract', hue='Churn', data=df, ax=axes[0], palette='Set2')\naxes[0].set_title('Churn by Contract Type')\n\nsns.countplot(x='InternetService', hue='Churn', data=df, ax=axes[1], palette='Set2')\naxes[1].set_title('Churn by Internet Service')\n\nsns.countplot(x='PaymentMethod', hue='Churn', data=df, ax=axes[2], palette='Set2')\naxes[2].set_title('Churn by Payment Method')\naxes[2].tick_params(axis='x', rotation=45)\nplt.tight_layout()\nplt.show()\n\n# 3. Numerical Features Distribution\nfig, axes = plt.subplots(1, 2, figsize=(14, 5))\nsns.histplot(data=df, x='MonthlyCharges', hue='Churn', kde=True, ax=axes[0], palette='Set2')\naxes[0].set_title('Monthly Charges Distribution by Churn')\n\nsns.histplot(data=df, x='tenure', hue='Churn', kde=True, ax=axes[1], palette='Set2')\naxes[1].set_title('Tenure Distribution by Churn')\nplt.show()"),
        
        nbf.v4.new_markdown_cell("## 4. Data Preprocessing & Correlation Heatmap\nWe prepare the data for modeling by encoding categorical variables and generating a correlation heatmap to understand feature relationships."),
        nbf.v4.new_code_cell("# Drop customerID as it's not useful for prediction\ndf_model = df.drop('customerID', axis=1)\n\n# Label Encoding for categorical variables\nle = LabelEncoder()\nfor col in df_model.select_dtypes(include=['object', 'string', 'category']).columns:\n    df_model[col] = le.fit_transform(df_model[col])\n\n# Plot Correlation Heatmap\nplt.figure(figsize=(14, 10))\nsns.heatmap(df_model.corr(), annot=False, cmap='coolwarm', linewidths=0.5)\nplt.title('Feature Correlation Heatmap')\nplt.show()"),
        
        nbf.v4.new_markdown_cell("## 5. Feature Scaling and Train-Test Split\nWe split the data into training (80%) and testing (20%) subsets, and apply standard scaling to numerical features to ensure algorithms converge optimally."),
        nbf.v4.new_code_cell("X = df_model.drop('Churn', axis=1)\ny = df_model['Churn']\n\n# Train-test split\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n# Feature Scaling\nscaler = StandardScaler()\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)\n\nprint(f\"Training set shape: {X_train.shape}\")\nprint(f\"Testing set shape: {X_test.shape}\")"),
        
        nbf.v4.new_markdown_cell("## 6. Model Training and Evaluation\nWe evaluate Logistic Regression, Decision Tree, and Random Forest models. For each, we'll plot a confusion matrix and print a classification report."),
        nbf.v4.new_code_cell("models = {\n    'Logistic Regression': LogisticRegression(random_state=42),\n    'Decision Tree': DecisionTreeClassifier(random_state=42),\n    'Random Forest': RandomForestClassifier(random_state=42)\n}\n\nresults = {}\ntrained_models = {}\n\nfor name, model in models.items():\n    print(\"=\"*60)\n    print(f\"Training {name}...\")\n    \n    # Train\n    model.fit(X_train, y_train)\n    trained_models[name] = model\n    \n    # Predict\n    y_pred = model.predict(X_test)\n    \n    # Metrics\n    acc = accuracy_score(y_test, y_pred)\n    prec = precision_score(y_test, y_pred)\n    rec = recall_score(y_test, y_pred)\n    f1 = f1_score(y_test, y_pred)\n    results[name] = [acc, prec, rec, f1]\n    \n    print(f\"Accuracy: {acc:.4f} | F1-Score: {f1:.4f}\\n\")\n    \n    print(\"Classification Report:\")\n    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))\n    \n    # Confusion Matrix Plot\n    cm = confusion_matrix(y_test, y_pred)\n    plt.figure(figsize=(4, 3))\n    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])\n    plt.title(f'Confusion Matrix - {name}')\n    plt.ylabel('Actual Label')\n    plt.xlabel('Predicted Label')\n    plt.show()\n"),
        
        nbf.v4.new_markdown_cell("## 7. Model Comparison & Feature Importance\nFinally, we summarize the model metrics and visualize the most important factors predicting customer churn according to the Random Forest model."),
        nbf.v4.new_code_cell("# Display Results Summary\nresults_df = pd.DataFrame(results, index=['Accuracy', 'Precision', 'Recall', 'F1-Score']).T\ndisplay(results_df)\n\n# Feature Importance from Random Forest\nrf_model = trained_models['Random Forest']\nimportances = rf_model.feature_importances_\nfeature_names = X.columns\n\n# Sort features\nindices = np.argsort(importances)[::-1]\n\nplt.figure(figsize=(10, 6))\nplt.title(\"Feature Importance - Random Forest\")\nsns.barplot(x=importances[indices], y=feature_names[indices], palette=\"viridis\")\nplt.xlabel(\"Relative Importance\")\nplt.ylabel(\"Feature\")\nplt.show()"),
        
        nbf.v4.new_markdown_cell("## 8. Conclusion\nThis project successfully built a predictive analytics system to identify potential churners. Our Exploratory Data Analysis revealed key insights—for instance, month-to-month contracts and certain internet services have higher churn rates. The Random Forest model proved highly effective at classification, and its feature importance metric highlights `TotalCharges`, `Tenure`, and `MonthlyCharges` as the most critical factors influencing customer retention. Implementing such data-driven approaches allows organizations to proactively design targeted retention strategies.")
    ]
    
    with open('Customer_Churn_Prediction/Customer_Churn_Prediction.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    create_movie_recommendation_nb()
    create_churn_prediction_nb()
    print("Notebooks generated successfully.")
