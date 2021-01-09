from models_evaluation.tools import make_table, make_table_rst

make_table(data_type="training")
make_table(data_type="training_noisy")

make_table(data_type="training_fine_tuned")
make_table(data_type="training_noisy_fine_tuned")

make_table(data_type="zero_shot")
make_table(data_type="zero_shot_fine_tuned")

make_table_rst(data_type="training")
make_table_rst(data_type="training_noisy")
make_table_rst(data_type="zero_shot")
make_table_rst(data_type="zero_shot_fine_tuned")
