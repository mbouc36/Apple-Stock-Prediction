# Project Overview
This project aims to predict the stock prices of Apple Inc. using historical data from the year 2000 to the present. By employing machine learning techniques and leveraging Python's data science libraries, we've developed a model that forecasts stock price movements with an emphasis on accuracy and reliability.

### Table of Contents
Installation
Data Source
Technologies Used
Methodology
Results
Usage
Contributing
License
Contact

### Installation
To set up this project locally, follow these steps:

Clone the repository to your local machine.
Ensure Python 3.8+ is installed.
Install the required dependencies:
Copy code
pip install -r requirements.txt

## Data Source
In this project, the stock price data for Apple Inc. was sourced from Investing.com. The dataset includes comprehensive historical data on stock prices, including opening price, closing price, high, low, volume, and percentage changes on a weekly basis from the year 2000 to the present. For further details or to access the data, please visit Investing.com.

## Technologies Used
- **Python**: For overall programming.
- **Pandas**: For data manipulation and analysis.
- **Numpy**: For numerical computations.
- **Scikit-learn**: For building and evaluating the machine learning model.
- **Matplotlib**: For visualizing the results.

## Methodology
The project follows these key steps:

1. Data Preprocessing: Cleaning and preparing the data for analysis.
2. Feature Engineering: Selecting and engineering features relevant to stock price predictions.
3. Model Building: Using Linear Regression as the primary predictive model.
4. Evaluation: Assessing the model's performance using SMAPE.

## Results
The model achieved a SMAPE of 12.67% on the training set and 21.53% on the cross-validation set, indicating its effectiveness in predicting Apple's stock prices with a reasonable degree of accuracy. For a detailed analysis, refer to the full report.

## License
This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

### GPL License Notice
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.

For more information on the GNU General Public License, please visit GNU's GPL How-to Guide.



