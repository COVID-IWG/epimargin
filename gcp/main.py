'''
Dummy function to test GCP functionality 
'''

import pandas as pd


def timenow(event, context):
    '''
    Function creates a csv file with timestamp
    
    Output:
        Returns a csv file
    '''
    import base64
    import pandas as pd
    
    print("""This Function was triggered by messageId {} published at {}
    """.format(context.event_id, context.timestamp))
    
    d= {"Time": pd.Timestamp.now()}
    
    df = pd.DataFrame(data=d, index=range(1))

    df.to_csv("/tmp/test_gcp.csv", index=False)
    
    print("Timestamp", d)


#if __name__ == "__main__":
#    timenow()
