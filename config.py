import os
os.environ["COMET_LOGGING_FILE"] = "out/comet.log"
os.environ["COMET_LOGGING_FILE_LEVEL"] = "debug"

# Create an experiment with your api key:
config = dict(
    api_key="e6sHTPWATwAE5PxBLojN1MuXH",
    project_name="general",
    workspace="jgeofil",
    auto_metric_logging=True,
    auto_param_logging=True,
    auto_histogram_weight_logging=True,
    auto_histogram_gradient_logging=True,
    auto_histogram_activation_logging=True,
    auto_histogram_epoch_rate=True,
    log_graph=True,
    auto_metric_step_rate=True,
    parse_args=True
)






