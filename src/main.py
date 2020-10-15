import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    customers = pd.read_csv('../data/ecommerce_customers.csv')
    toggle = False
    if toggle:
        print(customers.head())
        print(customers.info())
        print(customers.describe())

        plot1 = sns.jointplot(customers['Time on Website'], customers['Yearly Amount Spent'])
        plot1.savefig('../imgs/time_site_vs_yearly_spend.png')
        # Does this figure make sense? Not really, I'd expect positive linear relationship.

        plot2 = sns.jointplot(customers['Time on App'], customers['Yearly Amount Spent'])
        plot2.savefig('../imgs/time_app_vs_yearly_spend.png')
        # The app has a better positive linear relationship

        plot3 = sns.jointplot(customers['Time on App'], customers['Length of Membership'], kind='hex')
        plot3.savefig('../imgs/time_app_vs_len_member_hex.png')

        pairplot = sns.pairplot(customers)
        pairplot.savefig('../imgs/pairplot.png')
        # Based off this plot, length of membership looks to have the strongest correlation
        # weaker correlation between Time on App and Avg. Session Length

        lmplot = sns.lmplot(data=customers, x='Length of Membership', y='Yearly Amount Spent')
        lmplot.savefig('../imgs/lmplot.png')
    
    x_cols = ['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']
    y = customers['Yearly Amount Spent'].copy()
    X = customers[x_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    print(lm.coef_)
    # [25.98154972 38.59015875  0.19040528 61.27909654]
    y_pred = lm.predict(X_test)

    lm_results = sns.jointplot(y_test, y_pred)
    lm_results.set_axis_labels(xlabel='True Value', ylabel='Predicted Value')
    lm_results.savefig('../imgs/lm_results.png')

    mae = round(mean_absolute_error(y_test, y_pred), 2)
    mse = round(mean_squared_error(y_test, y_pred), 2)
    rmse = round(np.sqrt(mse), 2)

    print('Mean Absolute Error: ', mae)
    print('Mean Squared Error: ', mse)
    print('Root Mean Squared Error: ', rmse)

    '''
    Mean Absolute Error:  7.23
    Mean Squared Error:  79.81
    Root Mean Squared Error:  8.93
    '''
    residuals = y_test.values - y_pred
    residuals_plot = plt.hist(residuals, bins=35)
    #res_hist_plot.savefig('../imgs/residuals_hist_plot.png')
    
    coeffecients = pd.DataFrame(lm.coef_,X.columns)
    coeffecients.columns = ['Coeffecient']
    print(coeffecients)

    '''
                            Coeffecient
    Avg. Session Length     25.981550
    Time on App             38.590159
    Time on Website          0.190405
    Length of Membership    61.279097

    '''
        

    
    





