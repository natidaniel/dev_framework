import pandas as pd
import numpy as np
import time
start_time = time.time()
file_name="resnet-101-sup-face-0.8.csv"
full_file = pd.read_csv(file_name)
df_test = pd.DataFrame()
df_train = pd.DataFrame()
temp_ =[35.8,36,36.1,36.2,36.3,36.4,36.5,36.6,36.7,36.8,36.9,37,37.1,37.3,37.4,37.5]
for t in temp_:
    name_count = 0
    temp = full_file[full_file['Temp'] == t]
    count = temp.drop_duplicates(subset=["Unnamed: 0"]).count()[0]
    if (count==0):
        break
    test_count = int(0.2 * count)
    train_count = count - test_count
    name_list=temp['Unnamed: 0'].tolist()
    temp_name=name_list[0]
    for i, name in enumerate (name_list):
        if test_count==0:
            train_rows = temp.iloc[:]
            df_train = df_train.append(train_rows, ignore_index=True)
            break

        if name_count == test_count+1:
            test_rows = temp.iloc[0:i - 1, ]
            train_rows = temp.iloc[i - 1:, ]
            df_test = df_test.append(test_rows, ignore_index=True)
            df_train = df_train.append(train_rows, ignore_index=True)
            break


        if name == temp_name:
            continue
        name_count+=1
        temp_name=name

save_name=file_name.split('.csv')[0]
df_train.to_csv(save_name+'_train(0.6).csv')
df_test.to_csv(save_name+'_val(0.2).csv')



print("  %s seconds  " % (time.time() - start_time))