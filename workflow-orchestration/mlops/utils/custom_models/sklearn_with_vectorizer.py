import mlflow.pyfunc

class SklearnModelWithVectorizer(mlflow.pyfunc.PythonModel):
    def __init__(self, model, dv):
        self.model = model
        self.dv = dv

    def predict(self, context, model_input):
        X_transformed = self.dv.transform(model_input.to_dict(orient="records"))
        return self.model.predict(X_transformed)