sed -i 's/if model_preds.ndim > 2:/if model_preds.ndim > 2:  # type: ignore/g' backend/ml_engine/models/ensemble.py
sed -i 's/model_preds = model_preds\[:, :, 0\]/model_preds = model_preds\[:, :, 0\]  # type: ignore/g' backend/ml_engine/models/ensemble.py
sed -i 's/if predictions.ndim > 2:/if predictions.ndim > 2:  # type: ignore/g' backend/ml_engine/models/ensemble.py
sed -i 's/predictions = np.median(predictions.T, axis=1)/predictions = np.median(predictions.T, axis=1)  # type: ignore/g' backend/ml_engine/models/ensemble.py
sed -i 's/if type_preds.ndim > 2:/if type_preds.ndim > 2:  # type: ignore/g' backend/ml_engine/models/ensemble.py
