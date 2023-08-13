from pathlib import Path


class Config:
    """
    Configuration class for the project.
    """

    OUTPUT_PATH = Path("./output")
    RAW_PATH = OUTPUT_PATH / "raw"
    PREPROCESSED_PATH = OUTPUT_PATH / "preprocessed"
    RESULT_PATH = OUTPUT_PATH / "results"
    PLOT_PATH = OUTPUT_PATH / "plots"

    LINE_LENGTH = 120
