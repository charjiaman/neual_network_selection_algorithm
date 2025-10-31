'''
function: welfare_plot.py

create a seperate dataframe df_well to store the welfare where welfare = sum of R selected by prediction/sum of R from the original test labels.
plot welfare.
'''

import pandas as pd #import python data structure 
import seaborn as sns #plot purpose
import matplotlib.pyplot as plt #plot purpose    

def welfare_plt(df_predwx, df_testwx , cus_color, cus_title):

    df_well = pd.DataFrame()
    df_well['welfare'] = df_predwx['sum_r']/df_testwx['sum_r']#welfare is equal to prediction value/original ground truth
    #print(cus_title, '\n\n welfare value:\n', df_well)

    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    #plot welfare
    sns.histplot(data=df_well, x='welfare', bins=bins, color=cus_color, edgecolor='black', stat='percent')

    #calculate mean

    average_value = df_well['welfare'].mean()
    #representing the average
    plt.axvline(x=average_value, color='red', linestyle='dashed', linewidth=2, label=f'Average: {average_value:.2f}')

    #text annotation
    plt.text(average_value + 2, plt.gca().get_ylim()[1] * 0.9, f'Average: {average_value:.2f}', color='red')
    plt.title(cus_title)
    plt.xlabel('welfare range')
    plt.ylabel('percent')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(bins, [f'{bin:.1f}' for bin in bins])
    file_name = f"plot_{cus_title}.png"

    plt.legend()

    plt.savefig(file_name)  # Save as a PNG image
    plt.clf()
    #plt.show()

    return None