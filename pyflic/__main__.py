"""Entry point for ``python -m pyflic``."""


def main() -> None:
    print(
        "pyflic — FLIC data analysis toolkit\n"
        "\n"
        "Available commands:\n"
        "  pyflic-config   Launch the experiment config editor GUI\n"
        "  pyflic-qc       Launch the QC viewer  (usage: pyflic-qc <project_dir>)\n"
        "\n"
        "Python API:\n"
        "  from pyflic import load_experiment_yaml, Experiment\n"
    )


if __name__ == "__main__":
    main()
