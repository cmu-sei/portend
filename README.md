# portend-tools
Code for the Portend tools, a set of tools to help generate simulated drift on ML model data, and figure out proper metrics to detect it in operations.

## Contents

* [Portend Workflow](#portend-workflow)
* [Portend Tools](#portend-tools-1)
* [Setup](#setup)
* [Usage](#usage)
* [Development](#development)
* [Packaging and Installation](#packaging-and-installation)
* [Configuration Files](#configuration-files)
* [Embedding the Monitor](#embedding-the-monitor)
* [Extending](#extending)

## Portend Workflow

The use of Portend is divided in two main stages: Planning Stage and Operational Stage. Most of the tools are used in the Planning Stage.

The goal of the Planning Stage is to allow a model developer to find metrics that allow the early detection of drift in deployed models. This is done by generating different types of drifted data,
and then running different potential metrics to find which ones work better. The steps are summarized as follows:

1. Train the ML model. This is usually done outside Portend, though there is a Trainer tool that can help train in some cases.
1. Define and generate drifted data. This will take a set of normal data and generate one or more sets of drifted data. This is done using the Drifter tool.
   * This may also require implementing new drift generator functionality for the Drifter (see [extending section](#extending)).
1. Define metrics to test out, and run them in the Predictor tool.
   * This may require implementing new metrics, if not already available in Portend (see [extending section](#extending)).
   * This may also require implementing code for the ML model, if specific code is needed when loading or processing the outputs of the model (see [extending section](#extending)).
1. Analyze the results, and generate a configuration file for the Monitor. This is done in part manually, and through the Selector tool.

The goal of the Operational Stage is to be able to use the metrics so that they can be run while in operations. This is currently done by building the Portend package, and using the Monitor code inside it, by calling it from the operational code (specific to each deployment and system).

## Portend Tools

The following are the main tools:

* **Drifter**: a tool that can generate a drifted dataset from a given dataset. It can use multiple different drift generation libraries. Some are included with Portend, but it is easy to extend it to more classes by extending a defined interace and configuring the Drifter to use the new drift generators (see below for [extending information](#extending)). Multiple runs of the tool are needed to generate drifted datasets with different characteristics.

* **Predictor**: a tool that runs the model to a given (drifted) dataset, and calculate the configured metrics on it. The idea is to iterate on this while adding new metrics. The person using it has to analyze the results of the Predictor to find metrics that seem appropriate for the different types of drift.

* **Experiment Runner**: this is a collection of shell scripts, available at `configs/experiments`, that simplify running the Drifter and Predictor multiple times. By using a Helper Tool (see below) to generate slightly different drift generation config files, these scripts can run multiple experiments, generating drifted datasets with different drifts (such as slightly different variations of a parameter), and call the Predictor on the results of each one. This makes it easier to see trends in the metrics as the drift increases or decreases, and select the appropriate metric more easily. 

* **Selector**: a tool that takes an input the result of an experiment, and generates a configuration file to be used by the Monitor in an operational deployment. This config file will be a draft will all the metrics and recommended alert levels, and should be tweaked by the user to only include metrics that seem to be useful, and levels appropriate to the operational context.

* **Monitor**: a tool that currently works as a Python function, called from operational code, that indicates if drift is being detected. To be called every time the ML model in operations generates an output.

* **Helper**: this is a multi-purpose tool, optionally used to process data. It has three different actions:
  * "merge": merges two datasets into a unified JSON dataset file.
  * "image_json": generates a JSON dataset file compatible with the ImageDataSet class, from a folder with a set of image files
  * "config_gen": generates a set of JSON file configurations based on a template and a range of values, to help the Experiment Runner run multiple results with slightly different drift parameters.

* **Trainer**: another helper tool used to train a model, supporting only Keras models currently. Most models are expected to be trained outside Portend.

Most tools work by reading their configuration from a JSON configuration file, which is different for each tool. See [Configuration Files](#configuration-files) for more details.

## Setup

### Dockerized Setup

1. Docker is required.
   * If working behind a proxy, Docker needs to be configured to 1) download images and 2) so that images that are being built can download components from the Internet.
   * NOTE: external algorithms may not run properly inside the Dockerized version, depending on their specific dependencies.
1. From the project folder, run:
   * `bash build_container.sh`

### Local Environment Setup
1. Python 3.9 and pip3 are required.
1. Create a virtual environment with venv, and activate it, and install requirements:
   * `bash setup_local.sh`
1. (Optional) To set up tensorflow metal plugin on an Apple-M CPU:
   * Ensure your environment is activated: `source .venv/bin/activate`
   * `pip3 install '.[tfmac]'`


## Usage

### Drifter, Predictor, Selector, Helper, Trainer

All these tools are called by the same scripts, with different arguments.

#### Dockerized Usage

1. From the project folder, run:
   * `bash run_container.sh <tool_name> [params]`
   * Where 
      * <tool_name> is the name of one of the tools: `drifter`, `predictor`, `selector`, `helper`, `trainer`.
      * [params]: can include:
        * `-c`, followed by a config file name (path relative to the project folder), to indicate the configuration to use.
        * `--test`: runs the tool in test mode (if available).
        * `--pack`: for the Predictor, indicates the file prefix and the activation of the generation of a folder and a zip file in the output/packaged subfolder containing information about the run. More specifically, it contains 1) predictions JSON output, 2) metrics JSON output, 3) drifted ids JSON input, 4) trained model input, 5) log, 6) config file used.  
        * `--drifts`: for the Predictor, path to a set of drift config files used to generate the datasets being used, to be added to packaged zip file (only useful if `--pack` is used).
        * `--pfolder`: for the Predictor, path to an existing packaged folder to load the predictions form (avoids running ML model again). 
        * `--expfolder`: For the Selector, indidicates the folder with packaged folders from an experiment run to use to analyse the results from.

#### Local Environment Usage

1. Ensure your venv has been activated.
   * `source .venv/bin/activate`
1. From the project folder, run:
   * `bash run_local.sh <tool_name> [params]`
   * See [Dockerized Usage section](#dockerized-usage) for details on tool_name and params.

### Experiment Runner

This is a collection of scripts used to quickly run or re-run experiments. Experiments refer to the generation of drifted datasets with variations of drift, followed by running multiple metrics as well, packaging all results for easy review later. There are three main tools, all available at `config/experiments`, and it is expected to run them from that folder.

* ``exp_run.sh``: the main experiment runner script, generates multiple config files for the Drifter, runs it for each of them, and runs the Predictor for each drifter dataset, packaging all the results as well. Has the following signature: ``bash exp_run.sh <script_to_run> <exp_gen_config_path> <predictor_config_path>`` . Arguments:
   * "script_to_run": can only be `run_local.sh` or `run_container.sh`. Used to select whether to run the experiments in the local environment or in a container.
   * "exp_gen_config_path": path relative to `config\experiments` to JSON configuration file for Helper that indicates how to generate the Drifter configs to run.
   * "predictor_config_path": path relative to `config\experiments` to JSON configuration file for Predictor that will be used to run the model and metrics on each drifted dataset.

* ``re_run.sh``: script used to only run the Predictor again on already generated drifted datasets, created by an earlier run of ``exp_run.sh``. Useful to test new metrics when there is no need to re-generate the drifted data. This also avoids re-running the ML model again, instead loading its results from the packaged results of the previous experiment. This will result in much faster runs when only different metrics want to be tried on existing experiments. Has the following signature: ``bash re_run.sh <script_to_run> <exp_results_path>`` . Arguments:
   * "script_to_run": same as above.
   * "exp_results_path": path relative to this repo's root folder, to folder containing packaged experiments resulting from a previous run of ``exp_run.sh``

### Monitor

Currently the monitor can only be used in embedded mode, by importing the Portend package and calling the appropriate functions in the operational code. See the [Embedding the Monitor](#embedding-the-monitor) section for details.

## Development

To set up your local setup environment for local tools:
1. Set up the [Local Environment](#local-environment-setup) for Portend.
1. Activate the environment:
    * `source .venv/bin/activate`
1. Run the following command to install the needed dependencies to lint, test and build:
    * `pip3 install '.[dev]'`

To run development tools on your local setup:
* Run the following command to lint your code:
    * `make qa`
* Run the following command to run all unit tests:
    * `make test`

## Packaging and Installation
To create a package for Portend that can be distributed or installed, follow these steps.
1. Ensure all setup steps from the previous [Development](#development) section have been followed, and the env is still active.
1. Create the Portend package:
    * `python -m build`
1. The resulting package will be in the `./dist` subfolder, with the extension `.whl`.

To install the packaged Portend in another Python environment or device:
1. Build the packaged Portend in the target device, as described above, or, if compatible, copy the package with the `.whl` extension to the device you want to install it in.
1. Go to the environment you want to run Portend on.
1. Run the following command to install it in the currently active Python environment:
    * `pip3 install <path_to_whl_file>`

## Configuration Files

Each tool has a specific type of configuration file format. Several of the bash files here reflect different configurations for the same tool, to do different things. Below is the description of each config format, and the current bash files using them as well. The `configs/` folder has examples of several different configuration files. Note that if you are running the tools in Dockerized mode, you should put your configurations inside the one of the mounted folders (`configs`, `input`, `output`, `process_io`), so that they are available to the tools inside the Docker container.

### Common Sections
The following sections are common and can be used for the Trainer, Drifter and Predictor tool in the same way:
 - **datasets**: information about the input datasets. It is an array of dicts. Each dict has the following fields:
   - **dataset_file**: relative path to a JSON file with the labelled dataset to use for generating a drifted one.
   - **dataset_class**: (OPTIONAL) name of the dataset class extending DataSet and implementing dataset-specific functions. It has the format "<module_path>.<class_name>" (i.e., "portend.examples.iceberg.iceberg_dataset.IcebergDataSet"). If not provided, default `DataSet` class is used.
   - **dataset_id_key**: (OPTIONAL) name of the column/field to be used as identifier of the dataset.
   - **dataset_timestamp_key**: (OPTIONAL) name of the column/field to be used as timestamp.
   - **dataset_input_key**: (OPTIONAL) name of the column/field to be used as model input, in case one column is the only input.
   - **dataset_output_key**: (OPTIONAL) name of the column/field to be used as model output, in case one column is the only output.
   - **dataset_file_base**: (OPTIONAL) only needed if dataset_file is a refernce dataset, that references this base dataset.
 - **model**: information about the model. Only useful for Trainer and Predictor. Has the following subkeys:
   - **model_class**: name of the model class extending MLModel and implementing model-specific functions. It has the format "<module_path>.<class_name>" (i.e.,  "portend.examples.iceberg.iceberg_model.IcebergModel"). If not provided, default `KerasModel` class is used.
   - **model_file**: (OPTIONAL) model to load, if needed (for evaluation or prediction).
   - **model_command**: (OPTIONAL, needed if using `ProcessModel`) a string with the command to run the algorithm process and get predictions.
   - **model_working_dir**: (OPTIONAL, still optional if using `ProcessModel`) the working directory from which to run the algorithm command.
   - **model_input_folder**: (OPTIONAL, needed if using `ProcessModel`) the input folder where the input files will be moved into for the algorithm, relative to `./process_io/`. This should be used by the external algorithm should read input from (relative to `model_container_io_path` below if running in a container).
   - **model_output_file**: (OPTIONAL, needed if using `ProcessModel`) the output file where the algorithm process will output the CSV with predictions, relative to `./process_io/`. This should be used by the external algorithm should write outputs to (relative to `model_container_io_path` below if running in a container).
   - **model_extra_output_files**: (OPTIONAL, optional if using `ProcessModel`) a list of additional output files where the algorithm process will output additional data in CSV format, relative to `./process_io/`. This should be used by the external algorithm should write outputs to (relative to `model_container_io_path` below if running in a container).
   - **model_container_name**: (OPTIONAL, needed if using `ProcessContainerModel`) the name of a container image to run.
   - **model_container_params**: (OPTIONAL, still optional if using `ProcessContainerModel`) any additional params to pass to the container, if needed.
   - **model_container_io_path**: (OPTIONAL, needed if using `ProcessContainerModel`) an absolute path on the container to mount the process IO path into, so that it can receive its inputs and put its outputs.
   - **model_container_gpus**: (OPTIONAL) the GPU device parameters to select for assignining GPUs to the container. Can be "none" (default), "all" for all GPUS, or one or more number separated by commas to assign specific GPUs (see docker run gpus param for details).


### Drifter Tool Config

The Drifter tool configuration has the following fields:

 - **dataset**: information about the input dataset. See Common section above for details on what can go inside this section.
 - **output**: section for output files.
   - **output_dataset_file**: relative path to a JSON file that will contain the drifted dataset. In test mode, this is the dataset that will be tested.
   - **save_backup**: if present, will make the Drifter create a timestamped copy of the drifted dataset JSON file in the output folder.
 - **drift_scenario**: information about the drift being generated.
   - **condition**: friendly name for the drift type. 
   - **module**: full path of the Python module implementing the drift (built-ins are inside `portend.drifts`). Current options are:
     - "portend.drifts.temporal.temporal_drift": contains several types of temporal drift.
     - "portend.drifts.image.image_drift": contains several types of image drift, including fog and flood.
   - **params**: dictionary of specific parameters for this drift generator. Depend on the **module** defined above. Each module has some common params, and usually submodules have additional params to be added as well. Some common params include:
     - **submodule**: not mandatory, but needed for "temporal_drift" and "image_drift". Indicates which of the internal drifts to use. Current values for temporal drift include: **math_drift**, **prevalence_drift**, and **random_drift**. Current values for image drift include: **fog.fog** and **flood.flood**.


### Predictor Tool Config

The Predictor tool configuration has the following fields:

 - **mode**: can be "analyze", which will execute the model and output predictions and metrics, "predict" which only outputs the predictions, or "label", which will execute the model but write an updated labelled dataset with that output.
 - **threshold**: value between 0 and 1 that will convert a raw prediction into a classification.
 - **datasets**: information about the input datasets. See Common section above for details on what can go inside this section. Each dict has some additional fields:
     - **predictions_output**: relative path to JSON file where the predictions will be stored. Only needed in "analyze" and "predict" modes.
     - **labelled_output**: relative path to JSON file where the labelled output will be stored. Only needed in "label" mode. 
 - **model**: information about the model. See Common section above for details on what can go inside this section.
 - **time_series**: information about the time series, when used for analysis. Only needed in "analysys" mode.
     - **ts_model**: relative path to the folder where the trained time-series to be used is stored.
     - **time_interval**: time interval configuration.
        - **starting_interval**: time interval at which start aggregating the predictions for analyzing the results.
        - **interval_unit**: time interval unit. Possible values available here: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
 - **analysis**: section for analysis related configuration, only needed in "analyze" mode.
     - **packaged_folder**: (optional) folder to put packaged results, if --store argument is used when running tool (will default to output/packaged)
     - **metrics_output**: relative path to the JSON file where the metric information will be stored
     - **metrics**: array containing objects describing the metrics to analyze. Each metric object contains:
        - **name**: a friendly name for the metric.
        - **type**: (optional) the metric type (can only be Metric, ErrorMetric, DistanceMetric). Defaults to "Metric".
        - **metric_class**: full path of the Python class implementing this metric (built-ins are inside `portend.metrics`).
        - **params**: dictionary containing drift module specific parameters. Any metric-specific parameters can be used here, but there are some default ones:
          - **prep_module**: (Optional) full path/name to a module that will prepare the data to be used by the metric, if default results are not enough and special processing is needed. Must implement a function that returns whatever is needed for the specific metric.
          - **prep_function**: (Optional) full path/name to the function in prep_module that will execute the prep code. Defaults to "prep_metric_data".
          - For DistanceType metrics, it has to contain at least these parameters:
            - **distribution**: the distribution to use. Supported values are "normal" and "kernel_density". Another option is to use "custom" as a value, which means that the metric module will implement the actual density function (see general README for more details).
            - **range_start** and **range_end**: limits for the helper array of potential valid values for this distribution.
            - **range_step**: step for the helper array for the distribution.


### Selector Tool Config

This tool helps generating the configuration file for a monitor to be used in operations.

  - **monitor_config_output_file**: path to file where the Monitor configuration output should be stored.
 - **generator_config_file_prefix**: (optional) prefix of the file used to generate the drifted data.
 - **num_alert_levels**: (optional) a number greater or equal to 1, indicating how many alert levels should be configured by deault on the Monitor configuration file (defaults to 4).


### Helper Tool Config

This is a separate, simpler tool to run specific helper functions. Its configuration has the following fields:

 - **action**: currently supports "merge", "image_json", and "config_gen". The rest of the parameters in the config file depend on the action.

 For action "merge", which merges two datasets into a unified JSON dataset file:
 - **dataset1**: path to first JSON dataset to merge.
 - **dataset2**: path to second JSON dataset to merge.
 - **output**: path to JSON file with merged dataset.

For action "image_json", which generates a JSON file compatible with the ImageDataSet class, from a folder with a set of image files:
 - **image_folder**: path to a folder with images.
 - **json_file**: JSON file to create with the dataset information about the images in the folder, one sample per image.

For action "config_gen", which generates a set of JSON file configurations based on a template and a range of values:
 - **base_file**: the template or base JSON config file to use and create the new ones from.
 - **output_folder**: folder where all created config files will be put.
 - **fields**: array of dictionaries with fields to modify in the new config files.
    - **key**: a key to find in the template config, which will have it's value modified in the new configs.
    - **min**, **max**, **step**: the min, step, anx max to use for the range of values. One config file will be created for each step, from min to max.


### Trainer Tool Config

The Trainer tool configuration has the following fields:

 - **training**: only if "on", it will train a new model using **dataset** and store it in **model**.
 - **cross_validation**: if "on", it will perform k-fold cross validation with the given **dataset**, to compare the results of training with different subsets of the data. This will just present results in the log, and won't store a new model.
 - **evaluation**: if "on", this will load the model indicated in **model** and evaluate it, using the full **dataset** if **training** was off, or the data split for validation if **training** was on (so it was executed).
 - **time_series_training**: only if "on", it will train a time-series model using **dataset** and store it in **ts_model**.
 - **dataset**: information about the input dataset. See Common section above for details on what can go inside this section.
 - **model**: information about the model. See Common section above for details on what can go inside this section.
 - **output_model**: relative path to the folder where the trained model will be stored.
 - **hyper_parameters**: parameters used when training the model. Only used if **training** is "on".
   - **epochs**: number of epochs to use while training.
   - **batch_size**: size of each batch to use while training.
   - **validation_size**: % of the dataset to split for validation.

The following fields are only used if **time_series_training** is "on".
 - **ts_output_model**: relative path to the folder where the trained time-series model will be stored.
 - **ts_hyper_parameters**: parameters used when training the time-series model.
   - **order_p, order_q, order_d**: ARIMA training parameters.
 - **time_interval**: time interval configuration.
   - **starting_interval**: time interval at which start aggregating the data for creating the time-series and training its model.
   - **interval_unit**: time interval unit. Possible values available here: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
   - **samples_per_time_interval**:  (OPTIONAL) only needed if dataset does not have timestamp data. In this case, this is the number of data samples that will be aggregated by time interval (instead of using timestamps for this aggregation).


## Embedding the Monitor
The Monitor can be used as embedded code from a Python program. Portend should be [packaged and installed](#packaging-and-installation) in the Python environment that will use the Monitor first for it to be available. Once it is installed, it is used in this way:

* Import the monitor package:
  * `from portend.monitor.alerts import calculate_alert_level`
* Optionally, import the method to load a configuration from a file (if you won't provide it directly as a dict):
  * `from portend.monitor.config import load_config`
* Optionally, load the configuration from a file:
  * `config = load_configuration("path_to_config.json")`
* Call `calculate_alert_level` with the appropriate parameters.

The following is the signature and details of `calculate_alert_level`:
 * Signature: `def calculate_alert_level(sample: dict, result: Any, additional_data: dict[str, dict], config: dict) -> str:`
 * Return value: a string indicating an alert level (semantics of this string are domain-dependent, and should be consistent with configuration). The string "none" indicates that no alert was generated.
 * Arguments:
   * `sample`: a dictionary with the information the model would need to run on the sample.
   * `result`: whatever main output was produced by running the model on the sample.
   * `additional_data`: a dictionary where each key/value pair corresponds to additional data produced by the model, besides the output. This is used by some metrics.
   * `config`: a dictionary of configurations to indicate metrics, thresholds and alert levels, that looks like the sample below. If multiple metrics are used, they are executed in order, and the first one to find a value with an alert level that is not "none" will be selected. The alert levels need to be in order of most critical to least critical.
```
  {
    "metrics":
    [
      {
        "name": "ATC",
        "metric_class": "portend.metrics.atc.atc.ATCMetric",
        "params":
        {
          "prep_module": "portend.examples.uav.wildnav_prep",
          "additional_data": "confidences",
          "average_atc_threshold": 0.7
        }
      }
    ],
    "alerts": {
          "ATC_metric":[
              {"less_than": 0.85, "alert_level": "critical"},
              {"less_than": 0.95, "alert_level": "warning"}
          ]}
  }
```


## Extending

Portend can be extended to support datasets and models that require specific handling, or to add new drift generation functionalities and drift detection metrics.

### Dataset Structures
The default class `datasets.dataset.DataSet` can be used without any changes for simple datasets, indicating some minor configuration related to model inputs and outputs through config files for the different tools (see below for details on the config files). It assumes the dataset information will come in a JSON file.

The `datasets.dataset.ImageDataset` class that comes with Portend can be used for image-based datasets where image files are stored separately than the dataset JSON file.

If additional processing or non-standard things are needed for a dataset, a new class can be implemented that derives from `datasets.dataset.DataSet`, and that overrides the methods as needed.
 
The most common methods to use or override are:
 - `def post_process(self)`: if data needs to be post-processed after loading from file, this is the method to override. It is automatically called after loading from file. By default it does nothing.
 - `def get_model_inputs(self) -> list[Sequence[Any]]`: returns the inputs needed for the corresponding model. By default returns an array only with the configured column for input, if any. Can be overriden for more complex inputs.
 - `def get_model_output(self) -> Sequence[Any]`: returns an array with the model output values. By default returns only the configured column for output, if any. Can be overriden for more complex outputs.
 - `def as_dataframe(self) -> pd.DataFrame`: returns the dataset as a Pandas dataframe, main method to get data, usually doesn't need overriding.

### ML Models
There are several ML models supported by the system that can be used directly it there is no need to train a model:
 - `models.keras_model.KerasModel`: a Keras-based model loader/handler. Can also be used to train simple Keras models if needed.
 - `models.torch_model.TorchModel`: a Pytorch-based model loader/handler. Has to be extended if training inside Portend is required.
 - `models.process_model.ProcessModel`: a class to run models externally as external processes. This can be used to run any model as a separate executable (such as a Python script). There is a defined interface that needs to be implemented by the external process for input and output:
   - Input: files to be used as input are passed in a configurable input folder. The file paths will be obtained from the JSON file, from the column configured as model input.
   - Output: should be a CSV file with the predictions for each input in the same line, separated by commas.
   - Additional configuration options are available in the `model` section of the configuration file (see config files section below for details).
- `models.process_model.ProcessContainerModel`: a more specific case of the ProcessModel, it allows the same things and interfaces, but simplifies the config for running an external process algorithm inside a container. Adds specific config options to indicate image name, params, etc (see config files section below for details).   

If model extensions are needed, specific models should extend from any of the models above, or from `models.ml_model.MLModel`. Some examples are available in `portend/examples`.

A model extension can overwrite functions as needed. If extending from a derived from `MLModel`, there are no functions that are mandatory. If derived directly from `MLModel`, the following functions can or have to be implemented:
 - `def predict(self, input: Any) -> Sequence[Any]`: the function that will return predictions based on inputs.
 - (Optional) `def load_from_file(self, model_filename: str)`: only needed if models want to be loaded from files.
 - (Optional) `def load_additional_params(self, data: list)`: only needed if additional configuration options need to be loaded for the model from the main config file being used.

If the trainer tool is going to be used, more functions are to be implemented. Some of these are already implemented for exsiting models, such as `KerasModel`.
 - `def save_to_file(self, model_filename: str)`: to save a model to a file.
 - `def train(self, training_set: TrainingSet, config_params: dict[str, str])`: to train a model.
 - `def evaluate(self, evaluation_input: list, evaluation_output: list, config_params: dict)`: to evaluate a model.
 - `def create_model()`: creates a model object for this dataset, and it has to store it in `self.model` .
 - `def split_data(self, dataset: DataSet, validation_percentage: float) -> TrainingSet`: to split data for training.
 - `def get_fold_data(self, dataset: DataSet, train_index: int, test_index: int) -> TrainingSet`: to get fold data for evaluating.

### Drift Modules
New drift modules can be added to the `portend/drifts` folder, or to any other location. Each module only needs to have one function, and has one optional function, with the following signatures:

 - `def apply_drift(dataset: DataSet, params: dict) -> DataSet:` This function should return a drifted dataset given a dataset, and whatever additional params are needed for this algorithm. See `tools/README.MD` for details on the config file format and params. More specifically:
   - `dataset`: an object of type `datasets.dataset.DataSet`, or derived from it. It comes loaded with the dataset data from the current drifter configuration being used. Both the actual class and the JSON file with the data are defined in the main drifter config file.
   - `params`: a dictionary obtained from the config file for the drifter, with params specific for this drift algorithm.
 - `def test_drift(dataset: DataSet, params: util.Config):` This optional would be used to test out this drift, printing results about its behaviour.

### Metric Modules
New metric modules need to implement a class extending from one of the base metric classes: `portend.metrics.basic.BasicMetric`, `portend.metrics.ts_metrics.ErrorMetric`, `portend.metrics.ts_metrics.DistanceMetric`. It has to implement the following methods, depending on the metric type.

For Basic Metrics:
 - `def _calculate_metric(self) -> Any:`: Calculates the given metric. `self` has:
    - `self.predictions: list[Predictions]`: a list of one or more `Predictions` objects, with results of a model.
    - `self.datasets: Optional[list[DataSet]]`: a list of one or more `DataSet` objects, with the original datasets used for the model.
    - `self.config: dict[str, Any] = {}`: a dictionary of optional configuration parametesr.

For Distance Metrics:

 - `def metric_distance(self, p: npt.NDArray[Any], q: npt.NDArray[Any]) -> Any:`: Calculates a distance value between the two given probability distributions (numpy arrays).

For Error Metrics:

  - `def metric_error(self, time_interval_id: int, time_series: TimeSeries, ts_predictions: TimeSeries) -> Any:`: Calculates the error for the given time interval, knowing the aggregated data from the time_series, and the time series predictions as a parameter.

### Examples

Some extension examples are included in the `portend/examples` package, and sample configs in the `configs` folder.

#### Iceberg Example

This example uses data from the Kaggle Iceberg Classifier Challenge (https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge) as data that could be drifted and used in Portend. More specifically, it contains two sample files:

 - `iceberg/iceberg_dataset.py`: an example of extending the basic Dataset to do post-processing when loading a dataset, and to set up not-default inputs for a model. This matches the datasets used in the Kaggle challenge.
 - `iceberg/iceberg_model.py`: an example of extending by adding an explicit Tensorflow ML model. This model is an example from the Kaggle challenge.

There are also sevearal configuration files in `configs\iceberg` that show a somewhat standard flow of configs needed to work with the Kaggle challenge example.

#### Wildnav UAV

This example uses data from the Wildnav paper (https://github.com/TIERS/wildnav), though the actual version of Wildnav used here is a fork with some more flexibility in its inputs and outputs (https://github.com/sebastian-echeverria/wildnav). More specifically, it contains two sample files:

  - `uav/wildnav_dataset.py`: similar example of post-processing and changing inputs, but this post-processes additional data files which are generated by the Wildnav executable.
  - `uav/wildnav_prep.py`: sample `prep_metric_data` function that can be set in the [Predictor Tool Config](#predictor-tool-config). This is used to pre-process data before a model is loaded, in case data needs to be transformed, cleaned, etc, before the model will actually work with it. In this case, it extracts intermediate data from additional files generated by Wildnav, which is the actual data (confidence in this case) being evaluated by Portend.

There are also sevearal configuration files in `configs\uav` that show a somewhat standard flow of configs needed to work with the Wildnav example, as well as files in `configs\experiments\uav`, which contain configurations to use Wildnav in different locations, as well as scripts to download tiles to use as pictures and maps of those areas. 

Note that the `map_tools` folder also contains several tools to help get and setup data for this Widlnav UAV example, which are references by the scripts in the experiments folder to obtain and prepare the tiles.

Note tha the `tyler` folder has a tile server that can be used to serve tiles, even enabling some of the existing image drifts supported by the system.
