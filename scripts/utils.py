import pandas as pd
from pathlib import Path
import filelock
from copy import deepcopy


class ExperimentReader:
    def __init__(self, path, name, exp_params):
        self.path = Path(path)
        self.name = name
        self.exp_params = exp_params

    def check_done(self, exp_params):
        raise NotImplementedError

    def write_result(self, result_dict):
        raise NotImplementedError


class CSVExperimentReader(ExperimentReader):
    def __init__(self, path, name, exp_params):
        super(CSVExperimentReader, self).__init__(path, name, exp_params)
        self.csv_file = self.path / f"{self.name}.csv"
        self.filelock = filelock.FileLock(str(self.csv_file) + ".lock")
        self.exp_params: dict = exp_params
        for k in self.exp_params:
            self.exp_params[k] = str(self.exp_params[k])

    def check_done(self):
        csv = pd.read_csv(self.csv_file)
        keys = list(self.exp_params.keys())
        values = list(self.exp_params.values())

        print(keys)
        exp_params_csv = csv[keys]
        done = exp_params_csv.isin(values).all(axis=1).any()
        return done

    def write_result(self, result_dict):
        with self.filelock:
            if not self.csv_file.exists():
                self.csv_file.touch()

                df = pd.DataFrame(columns=list(self.exp_params.keys()))
                df.to_csv(self.csv_file, index=False)
            csv = pd.read_csv(self.csv_file, dtype=dict([k, str] for k in self.exp_params.keys()))
            exp_params_csv = csv[list(self.exp_params.keys())]
            exp_index = (exp_params_csv == pd.Series(self.exp_params)).all(axis=1)
            if not exp_index.any():
                r_dict = deepcopy(self.exp_params)
                r_dict.update(result_dict)

                new_df = pd.DataFrame(r_dict, index=[0])
                csv = pd.concat([csv, new_df], ignore_index=True)
                csv.to_csv(self.csv_file, index=False)
            else:
                csv.loc[exp_index, list(result_dict.keys())] = list(result_dict.values())
                csv.to_csv(self.csv_file, index=False)

    def read_result(self, target_keys):
        with self.filelock:
            csv = pd.read_csv(self.csv_file, dtype=dict([k, str] for k in self.exp_params.keys()))
            exp_params_csv = csv[list(self.exp_params.keys())]
            exp_index = (exp_params_csv == pd.Series(self.exp_params)).all(axis=1)
            if not exp_index.any():
                raise ValueError("Experiment not found")
            else:
                if not isinstance(target_keys, list):
                    target_keys = [target_keys]
                try:
                    result = csv.loc[exp_index, target_keys].iloc[0].to_dict()
                except KeyError:
                    result = None
        return result


if __name__ == "__main__":
    tst = CSVExperimentReader("./", "test", {"a": 1, "b": 2})
    print(tst.check_done())
    tst.write_result({"e": 3})
    tst.write_result({"e": 4})
    print(tst.check_done())

    tst = CSVExperimentReader("./", "test", {"a": 1, "b": 4})
    print(tst.check_done())
    tst.write_result({"f": 4})
    print(tst.check_done())
