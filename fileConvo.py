#using geeks for geeks for the quick conversion
import pandas as pd 
 
tsv_file='contentData.tsv'
 
# reading given tsv file
csv_table=pd.read_table(tsv_file,sep='\t')
 
# converting tsv file into csv
csv_table.to_csv('contentData.csv',index=False)
 
# output
print("Successfully made csv file")