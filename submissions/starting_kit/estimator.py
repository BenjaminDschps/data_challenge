# %load submissions/starting_kit/estimator.py

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer

cols = ['TIME_PERIOD', 'GEO', 'number_courses', 'job_offer',
       'need_for_manpower', 'difficult_recruitment', 'out_of_list',
       'entry_on_list', 'population']

categorical_cols = ['GEO', 'TIME_PERIOD']
numerical_cols = ['number_courses', 'job_offer',
       'need_for_manpower', 'difficult_recruitment', 'out_of_list',
       'entry_on_list', 'population']

transformer = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('passthrough', numerical_cols)
)

def get_estimator():
    pipe = make_pipeline(
        transformer,
        SimpleImputer(strategy='most_frequent'),
        LinearRegression()
    )

    return pipe