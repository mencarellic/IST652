import pandas as pd
import random
import pymongo
#import dnspython

# Show all columns and do not truncate in a DF
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', -1)

# Connect to the mongoDB instance
conn = pymongo.MongoClient("mongodb+srv://access:nhk35zjuKQATrmum@ist652-sd5uf.mongodb.net/test?retryWrites=true")


def mongo_to_df(connection, database, collection, n=0, rand=False, query={}):
    # Try to connect to the database and collection
    try:
        db = connection[database]
        coll = db[collection]
    except Exception as e:
        exit(e)

    # Get number of documents in collection for the query
    count = coll.count_documents(filter=query)

    # If the result has 0 documents, exit
    # Else print number found
    if count == 0:
        exit('Query matched no documents')
    else:
        print('Found {:d} documents.\n'.format(count))

    # Assign the query results to variable
    results = coll.find(query)

    # If random=True and n > 0
    if (rand) and (isinstance(n, int)) and n > 0:
        print('Capturing {:d} results in random order into a DataFrame.\n'.format(n))
        df = pd.DataFrame(list(results)).sample(n=n)
    # If random=True and n = 0
    elif (rand) and (n == 0):
        print('Capturing {:d} results in random order into a DataFrame.\n'.format(count))
        df = pd.DataFrame(list(results)).sample(n=count)
    # If random=False and n > 0
    elif (isinstance(n, int)) and n > 0:
        print('Capturing {:d} results in sequential order into a DataFrame.\n'.format(n))
        df = pd.DataFrame(list(results))[:n]
    # If n and random arent defined
    else:
        print('Capturing {:d} results in sequential order into a DataFrame.\n'.format(count))
        df = pd.DataFrame(list(results))

    # drop the mongodb id column
    df.drop(axis=1, columns=['_id'], inplace=True)

    # remove new line characters (\r \n)
    df.replace(r'(\r|\n|\r\n)', ' ', regex=True, inplace=True)

    # remove all extra spaces (\s)
    df.replace(r'(\s)+', ' ', regex=True, inplace=True)

    return df

thebeatles = mongo_to_df(connection=conn, database='Project', collection='songdata',
                         query={'artist': 'The Beatles'}, n=25, rand=True)

print(thebeatles.head(10))

