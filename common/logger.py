# Logger to be used while training to log details #

import numpy as np
import mlflow
import time

class MLFlowLogger:
    def __init__(self, uri = None, experiment_name = None, run_name = None):
        self.uri = uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self._intialize_mlflow()

    def _intialize_mlflow(self):
        mlflow.set_tracking_uri(uri=self.uri)
        mlflow.set_experiment(self.experiment_name)

    def start(self):
        self.run = mlflow.start_run(run_name=self.run_name)
        return self.run
    
    def end(self):
        if self.run:
            mlflow.end_run()
            self.run = None

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metric(self, name, value, step):
        mlflow.log_metric(name, value, step)
        