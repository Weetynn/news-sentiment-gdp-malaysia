# Leveraging News Sentiment to Forecast GDP's Demand Components: A Malaysian Case Study

## 📚 Introduction
This study examines the role of Mandarin news sentiment in forecasting Malaysia's GDP and its demand-side components - private investment, private consumption, imports, and exports. Given the delays in publishing key economic indicators, the research explores alternative methods for capturing economic sentiment through the analysis of Mandarin news articles. By analyzing Mandarin news articles and applying machine learning models, this study aims to provide more timely and accurate forecasts of Malaysia’s GDP and its key economic components.

## 📋 Critical Discussion Points

### CHAPTER 1: INTRODUCTION

#### 📌 Problem Statements

    ▪️ Macroeconomic indicators like GDP are reported with a 90-day delay in Malaysia, limiting timely decision-making.
    
    ▪️ The COVID-19 pandemic highlighted the need for real-time economic insights, as traditional indicators like GDP were too slow to capture rapid economic changes.
    
    ▪️ Survey-based indices like Business Condition Index (BCI) and CSI (Consumer Sentiment Index) often provide incomplete or biased data during economic downturns due to low response rates.

    ▪️ Most research focuses on English-language news sentiment, overlooking non-English sources like Mandarin, which is significant in Malaysia's multilingual context.
    
    ▪️ The study focuses on the predictive power of Mandarin news sentiment for forecasting Malaysia’s GDP, due to time and resource constraints.
    
    ▪️ Although high-frequency indicators are used in nowcasting, the study applies them to a quarterly forecasting task due to the available data.


#### 📌 Project Questions

The questions that this study aims to address are outlined as follows: 

    ▪️ Does the news sentiment index which is computed in the study accurately reflect the BCI and CSI published by the MIER?
    
    ▪️ How was the performance of the Mandarin text-derived news sentiment when compared to the English text-derived news sentiment in the work of Chong et al. (2021)?
    
    ▪️ Which of the four demand-side components (private investment, private consumption, imports, and exports) of GDP exhibited a strong correlation with the newly constructed news sentiment index, and which of the components showed a weak correlation to the index?

##### ⚠️ Note: The work of Chong et al. (2021) could be found [here](https://www.bis.org/ifc/publ/ifcb57_17.pdf). 


#### 📌 Aim and Objectives 

#### Aim:

    To assess the role of Mandarin news sentiment in forecasting Malaysia's GDP and its four demand components: private investment, private consumption, imports, and exports using machine learning techniques.
    

#### Objectives: 

    ▪️ To evaluate the effectiveness of Mandarin news sentiment in predicting Business Confidence Index (BCI) and Consumer Sentiment Index (CSI).
    
    ▪️ To compare the predictive performance of Mandarin news sentiment with English news sentiment as reported by previous studies.
    
    ▪️ To determine the correlation between GDP's demand-side components and the constructed news sentiment index.
    

#### 📌 Scope of the Study 

    ▪️ Macroeconomic indicators like GDP are reported with a 90-day delay in Malaysia, limiting timely decision-making.
    
    ▪️ The COVID-19 pandemic highlighted the need for real-time economic insights, as traditional indicators like GDP were too slow to capture rapid economic changes.
    
    ▪️ Survey-based indices like Business Condition Index (BCI) and CSI (Consumer Sentiment Index) often provide incomplete or biased data during economic downturns due to low response rates.

    ▪️ Most research focuses on English-language news sentiment, overlooking non-English sources like Mandarin, which is significant in Malaysia's multilingual context.
    
    ▪️ The study focuses on the predictive power of Mandarin news sentiment for forecasting Malaysia’s GDP, due to time and resource constraints.
    
    ▪️ Although high-frequency indicators are used in nowcasting, the study applies them to a quarterly forecasting task due to the available data.
    

#### 📌 Significance of the Study 

    ▪️ Fills gap in sentiment analysis, focusing on Mandarin in multilingual, multiethnic contexts.
    
    ▪️ Provides a benchmark for research in other multilingual economies on economic sentiment.
    
    ▪️ Enhances local forecasting of macroeconomic variables using news sentiment in Malaysia.

    ▪️ Aims to help policymakers improve economic trend forecasts for Malaysia's specific dynamics.


---

### CHAPTER 2: LITERATURE REVIEW

Provides a foundation for understanding the scope of the study and highlights existing research gaps. It covers four main areas:

    ▪️ Adoption of news sentiment for forecasting macroeconomic indicators like GDP growth, inflation, and unemployment.
    
    ▪️ Sentiment analysis of vernacular languages with a focus on Chinese, particularly in economic and financial contexts.
    
    ▪️ Application of machine learning models in sentiment analysis and their effectiveness in economic forecasting.
    
    ▪️ A review of the work by Chong et al. (2021), which forms the basis for this study.

#### 📌 Forecasting Key Economic Indicators through News Sentiments

    ▪️ News sentiment has been widely used for forecasting GDP, inflation, and unemployment.
    
    ▪️ There is a need for domain-specific sentiment analysis tools, as generic models may not capture economic nuances effectively.
    
    ▪️ Researchers have recommended broader exploration of big data indicators and textual data, especially in multiple languages.
    

#### 📌 Multilingual Sentiment Analysis (Emphasis on Chinese Texts)

    ▪️ There is significant work in sentiment analysis across various languages, but economic and financial texts in Chinese remain underexplored.
    
    ▪️ Studies have recommended culturally sensitive models and advanced techniques for real-time analysis, especially for platforms like Weibo.
    
    ▪️ Financial sentiment analysis in Chinese is scarce, highlighting the need for further research in this domain.


#### 📌 Review of Machine Learning Models for Sentiment Analysis

    ▪️ Machine learning models like Random Forest, Gradient Boosting, and LASSO show superior performance in economic forecasting.
    
    ▪️ Ensemble methods often yield more accurate results than individual models, especially for non-linear relationships and high-dimensional data.

    
#### 📌 Chong et al. (2021) and News Sentiment for Macroeconomic Forecasting

    ▪️ Chong et al.’s work, which focuses on using news sentiment for economic forecasting in Malaysia, serves as a benchmark for this thesis.
    
    ▪️ Their research emphasizes the importance of using multiple lexicons and advanced techniques for sentiment scoring and economic forecasting.

---

### CHAPTER 3: PROJECT METHODOLOGY 

The study follows the Cross-Industry Standard Process for Data Mining (CRISP-DM), which includes six stages: Business Understanding, Data Understanding, Data Preparation, Modelling, Evaluation, and Deployment.

The first stage of Business Understanding has been previously addressed in Chapter 1, covering the aims and objectives of the study. The deployment stage will not be included in this study, and details of the remaining four stages — Data Understanding, Data Preparation, Modelling, and Evaluation—will be further elaborated in Chapter 4.

---

### CHAPTER 4: IMPLEMENTATION

#### 📌 Data Collection Process

    ▪️ MIER Dataset: BCI and CSI data for 2022-2023 were purchased from the Malaysian Institute of Economic Research (MIER).
    
    ▪️ Macroeconomics Dataset: Data for GDP, private consumption, private investment, imports, and exports were sourced from the Department of Statistics Malaysia (OpenDOSM).
    
    ▪️ News Articles: A total of 3,361 articles from See Hua Daily News were scraped using ParseHub due to limitations with other news portals.


#### 📌 Initialization of the Data Analysis Process

    ▪️ Data Loading: Collected datasets (BCI, CSI, macroeconomic indicators, news articles) were loaded into Visual Studio Code for processing.


#### 📌 Initial Exploratory Data Analysis of the Textual Dataset

    ▪️ Exploration and Cleaning: The structure of the news articles dataset was explored, missing data was cleaned, and irrelevant columns were removed.
    
    ▪️ Preliminary Dataset Structure: The data was reviewed after preliminary cleaning to confirm its readiness for further analysis.


#### 📌 Initial Exploratory Data Analysis of the Numerical Datasets

    ▪️ Exploration of BCI, CSI, and Macroeconomic Data: The structure of BCI, CSI, and macroeconomic datasets was examined. Outliers were checked, and summary statistics were generated for further analysis.
    









    




