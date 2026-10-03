import os
from pathlib import Path

import io
import torch
import pickle


def bold_extreme_values_table_vis(data, format_string="%.3g", max_=True):
    data = data.astype(float).round(3)
    if max_:
        extrema = data != data.max()
    else:
        extrema = data != data.min()
    bolded = data.apply(lambda x: "\\textbf{%s}" % format_string % x)
    formatted = data.apply(lambda x: format_string % x)
    return formatted.where(extrema, bolded)


def to_str_table_vis(data, format_string="%.3g"):
    formatted = data.apply(lambda x: format_string % x)
    return formatted


def print_models(base_path, model_string):
    print(model_string)

    for i in range(80):
        for e in range(50):
            exists = Path(
                os.path.join(
                    base_path,
                    f"models_diff/prior_diff_real_checkpoint{model_string}_n_{i}_epoch_{e}.cpkt",
                )
            ).is_file()
            if exists:
                print(
                    os.path.join(
                        base_path,
                        f"models_diff/prior_diff_real_checkpoint{model_string}_n_{i}_epoch_{e}.cpkt",
                    )
                )
        print()


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Manager":
            from settings import Manager

            return Manager
        try:
            return self.find_class_cpu(module, name)
        except:
            return None

    def find_class_cpu(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)
