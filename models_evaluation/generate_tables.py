import os

from models_evaluation.tools import make_table, make_table_rst

# actual table
root_path = os.path.join(".", "results", "actual")
make_table(data_type="training", root_path=root_path)
make_table(data_type="training_incomplete", root_path=root_path)
make_table(data_type="zero_shot", root_path=root_path)

make_table_rst(data_type="training", root_path=root_path)
make_table_rst(data_type="training_incomplete", root_path=root_path)
make_table_rst(data_type="zero_shot", root_path=root_path)

# new table
root_path = os.path.join(".", "results", "new")
make_table(data_type="training", root_path=root_path)
make_table(data_type="training_incomplete", root_path=root_path)
make_table(data_type="zero_shot", root_path=root_path)

make_table_rst(data_type="training", root_path=root_path)
make_table_rst(data_type="training_incomplete", root_path=root_path)
make_table_rst(data_type="zero_shot", root_path=root_path)
