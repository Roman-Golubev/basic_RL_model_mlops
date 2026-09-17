import os
import pandas as pd

from scripts.rl_model import RL_func
from scripts.pipeline_config import DATA_INPUT_DIR, DATA_OUTPUT_DIR, API_CONFIGS, API_NUM


def bootstrap_calculation(api_num=API_NUM):
    cfg = API_CONFIGS[api_num]
    input_path = os.path.join(DATA_INPUT_DIR, cfg["params_file"])
    output_dir = os.path.join(DATA_OUTPUT_DIR, cfg["output_dir"])
    output_path = os.path.join(output_dir, cfg["output_file"])

    params_df = pd.read_csv(input_path)
    csv_path = os.path.join(DATA_INPUT_DIR, "fins.csv")
    fins_df = pd.read_csv(csv_path)
    output_df = pd.read_pickle(output_path)

    n_episodes = 20
    # продолжительность каждой итерации - один эпизод
    for _ in range(n_episodes):
        # в расчёте используется обученная Q_s_a
        results_dict = {
            'Q_s_a': None, 'ret_q': None, 'Pus': None, 'tem_distr': None,
            'fin_num': None, 'step_num_final': None,
            'states_lst': None, 'actions_lst': None, 'rewards_lst': None, 'del_Pus_lst': None
        }
        (
            results_dict['Q_s_a'], results_dict['ret_q'], results_dict['Pus'], results_dict['tem_distr'],
            results_dict['fin_num'], _, results_dict['step_num_final'],
            results_dict['states_lst'], results_dict['actions_lst'], results_dict['rewards_lst'], results_dict['del_Pus_lst']
        ) = RL_func(
            params_df.iloc[0], fins_df,
            alpha = 0.2,
            gamma = 1,
            epsilon = 0.7,
            max_steps = 10,
            n_episodes = 1,
            seed = 42,
            epsilon_decay={'decay_rate': 0.992, 'min_epsilon': 0.2},
            Q = output_df.iloc[-1, 0]
        )

        # сохранение результатов теущей итерации
        cur_result = pd.DataFrame([results_dict])
        output_df = pd.concat([output_df, cur_result], ignore_index=True)
        output_df.to_pickle(output_path)


if __name__ == "__main__":
    bootstrap_calculation()
