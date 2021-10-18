# CS596-Forex-Analysis

This project was completed as the Final Project for the Machine Learning course at San Diego State University (CS-596) for the Fall 2020 semester.

This project was completed by myself and my partner Nhat Ho. Since it was during COVID-19, it was difficult for us to meet and work on the analysis methods simultaneously. Therefore, we chatted over discord and decided to each take our own routes to analyzing the foreign exchange dataset we chose.

The code provided is entirely from my own portion of the project, and models A through E described in the report are all from my portion of the program as well. Model F was produced by my partner. 

The excel file used contains the daily foreign currency exchange rate between different currencies and the United States dollar. I chose 5 of these currencies to work with and generate regression models from. These were: the Euro, the Japanese Yen, the Chinese Yuan, the Indian Rupee, and the Australian Dollar.

The purpose of my linear regression models is to predict the future of a currency exchange rate based on the current and previous rates. The monthly model predicts the final day of the month based on the first 10 days, the bi-monthly model predicts the final day of the next month based on the first 15 days of a month, and the 200 days model predicts the value at the end of 200 days based on the first 100 days.

The purpose of my logistic regression model is to attempt to predict if a rate will rise or fall in the end of the period based on data from the beginning of the period. The 3 models generated use the same time periods as the linear regression model (monthly, bi-monthly, 200 days).

These models were fit and tested with portions of the datasets on each of the 5 currency relations with the US dollar. The purpose of these models is too attempt to provide a useful prediction for a person who wants to invest in one of the currency exchange rates and observe where the rate will be in the future or if it will go up or down based on historical data. The results are described in the project report. 
