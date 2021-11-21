import pathlib
import pandas as pd

from nn import nn_prediction

pathlib.Path('prediction').mkdir(parents=True, exist_ok=True)

prediction = nn_prediction(showEval=False, epochs=100, splt=False)

submission = {
    'ImageID': range(1, len(prediction)+1),
    'Label': prediction    
}

submission_df = pd.DataFrame(submission)

submission_df.to_csv('prediction/submission.csv', index=False)