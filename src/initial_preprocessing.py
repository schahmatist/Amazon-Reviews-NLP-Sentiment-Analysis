import numpy as np
import pandas as pd

df_raw=pd.read_csv('../data/raw/Reviews.csv.gz', compression='gzip')

#Checking for nulls

print('Total number or reviews:', len(df_raw))
print("Total number of na (nulls):", df_raw.isna().sum().sum())

print(f"\nDropping {df_raw.isna().sum().sum()} out of {len(df_raw)} reviews")
df_raw=df_raw.dropna()



# Defining Target dropping Neutral reviews:

df=df_raw[df_raw["Score"]!=3]
df['Updated_Score']=np.where(df['Score'].isin([1,2]), 'neg', df['Score'] )
df['Updated_Score']=np.where(df['Updated_Score'] !='neg', 'pos', df['Updated_Score'] )

#print('Original_number of rows:',len(df_raw), '\nUpdated number of rows: ', len(df))
print('\n-------------------------------------------------------------------------\n')

# selecting required columns:

print("Original columns: ", df.columns)

print("\nLeaving 'Text','Summary' and 'Score' columns:\n")
df=df[['Text','Summary','Score', 'Updated_Score']].copy()

print("Selected columns: ",df.columns)
print('\n-------------------------------------------------------------------------\n')



