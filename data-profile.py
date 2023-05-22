import pandas as pd
from pandas_profiling import ProfileReport

DATA_PATH = "dataset/"
FEATURE = "AAC"

train_neg = pd.read_csv(DATA_PATH + FEATURE + "_TR_neg_SPIDER.csv")
train_pos = pd.read_csv(DATA_PATH + FEATURE + "_TR_pos_SPIDER.csv")
test_neg = pd.read_csv(DATA_PATH + FEATURE + "_TS_neg_SPIDER.csv")
test_pos = pd.read_csv(DATA_PATH + FEATURE + "_TS_pos_SPIDER.csv")

train_frames = [train_neg, train_pos]
test_frames = [test_neg, test_pos]

train_df= pd.concat(train_frames)
test_df = pd.concat(test_frames)

profile = ProfileReport(train_df)

profile.to_file("report.html")