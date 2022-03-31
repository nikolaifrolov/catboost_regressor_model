from model import CatBoostModel
from preprocessing import PrepareData
from settings import *


if __name__ == '__main__':
    train_data, test_data, train_target = PrepareData(FILE_NAME, CAT_FEATURES, NUM_FEATURES, TARGET).prepare_data()
    model = CatBoostModel(train_data, train_target, test_data, MODEL_PARAMS, plot=False)
    model.train()
    df = model.predict()
    df.to_csv(OUTPUT_FILE_NAME, index=False)
    # model.save()
    # model.load()