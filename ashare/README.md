#### README #### 

The objective is to predict the electricity consumption for more than 1,500 buildings, for different energy uses annotated between 0 and 3 (explained). It refers to the kaggle competition for more details and to download data follow the link below: https://www.kaggle.com/c/ashrae-energy-prediction

public score: RMSLE=1.07

Analysis is made with 6 files including -2 files exclusively for train : train.csv (which contains the target) and weather_train.csv -2 files exclusively for test : test.csv, weather_test.csv ; -building_metadata.csv which describe buildings. Buildings are the same in train and test datasets. -sample_submission.csv to make the submission

building_metadata.csv describes buildings and containes data below: -site_id -building_id -primary_use -square_feet -year_built -floor_count

weather_train.csv -site_id -timestamp -air_temperature -cloud_coverage -dew_temperature -precip_depth_1_hr -sea_level_pressure -wind_direction -wind_speed

weather_test.csv -site_id -timestamp -air_temperature -cloud_coverage -dew_temperature -precip_depth_1_hr -sea_level_pressure -wind_direction -wind_speed

train.csv -meter_reading -building_id -meter -timestamp

test.csv -row_id -building_id -meter -timestamp

These buildings are distributed in different geographic areas (site id)

This study proposes to predict 2 years of consumption time series for 1500 buildings on python from 1 year of training data. I carried out the following main steps

-Minimization of the allocated memory. -Interpolation of missing data. -Feature Engineering -Forecast of time series with LightGBM with 5 folds cross validation.

The best predictions are for meter type 1 (rmse=0.85) and the worst for meter type 0 (rmse=1.16)

In the next step I should play with hyperparameters of gbm light for training of data for meter 0 type .



