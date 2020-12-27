from models_evaluation.tools import make_table, make_table_rst

make_table(data_type="training")
make_table(data_type="zero_shot")

make_table_rst(data_type="training")
make_table_rst(data_type="zero_shot")
