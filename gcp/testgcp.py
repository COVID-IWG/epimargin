'''
Dummy function to test GCP functionality 
'''

import pandas as pd


def timenow():
    '''
    Function creates a csv file with timestamp
    
    Output:
        Returns a csv file
    '''
    d= {"Time": pd.Timestamp.now()}
    
    df = pd.DataFrame(data=d, index=range(1))

    return df.to_csv("test_gcp.csv", index=False)


if __name__ == "__main__":
    timenow()
