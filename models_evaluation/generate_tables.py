import os

from models_evaluation.tools import make_table, make_table_rst, make_comparison_table

# actual table
root_path = os.path.join(".", "results", "actual")
make_table(data_type="training", root_path=root_path)
make_table(data_type="training_incomplete", root_path=root_path)
make_table(data_type="zero_shot", root_path=root_path)

make_table_rst(data_type="training", root_path=root_path)
make_table_rst(data_type="training_incomplete", root_path=root_path)
make_table_rst(data_type="zero_shot", root_path=root_path)

# comparison table
root_path = os.path.join(".", "results", "new")
results_a_file_name = "fasttext_256.json"
results_b_file_name = "fasttext_512.json"
make_comparison_table(results_a_file_name=results_a_file_name,
                      results_b_file_name=results_b_file_name,
                      root_path=root_path)
