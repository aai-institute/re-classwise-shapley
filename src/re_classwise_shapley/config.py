from pathlib import Path


class Config:
    """
    Configuration class for the project.
    """

    OUTPUT_PATH = Path("./output")
    RAW_PATH = OUTPUT_PATH / "raw"
    PREPROCESSED_PATH = OUTPUT_PATH / "preprocessed"
    SAMPLED_PATH = OUTPUT_PATH / "sampled"
    VALUES_PATH = OUTPUT_PATH / "values"
    RESULT_PATH = OUTPUT_PATH / "results"
    PLOT_PATH = OUTPUT_PATH / "plots"
