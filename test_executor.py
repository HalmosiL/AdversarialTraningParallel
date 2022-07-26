from Executor import Executor

Executor(
    model_cache="./model_cache/",
    max_number_of_gens=1,
    queue_size_train=10,
    queue_size_val=10,
    data_queue="./data_queue/",
    data_path="../../../../Data/City_scapes_data/",
    batch_size=1,
    device="cuda:0",
    number_of_steps=3,
    data_set_start_index_train=0,
    data_set_end_index_train=5,
    data_set_start_index_val=0,
    data_set_end_index_val=5
).start()