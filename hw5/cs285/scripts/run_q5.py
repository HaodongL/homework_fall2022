import shlex, subprocess


# python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 \
# --exp_name q5_easy_supervised_lam{}_tau{} --use_rnd \
# --num_exploration_steps=20000 \
# --awac_lambda={best lambda part 4} \
# --iql_expectile={0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99}
# python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 \
# --exp_name q5_easy_unsupervised_lam{}_tau{} --use_rnd \
# --unsupervised_exploration \
# --num_exploration_steps=20000 \
# --awac_lambda={best lambda part 4} \
# --iql_expectile={0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99}

# python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 \
# --exp_name q5_iql_medium_supervised_lam{}_tau{} --use_rnd \
# --num_exploration_steps=20000 \
# --awac_lambda={best lambda part 4} \
# --iql_expectile={0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99}
# python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 \
# --exp_name q5_iql_medium_unsupervised_lam{}_tau{} --use_rnd \
# --unsupervised_exploration \
# --num_exploration_steps=20000 \
# --awac_lambda={best lambda part 4} \
# --iql_expectile={0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99}


command_stem = [
"python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0   --use_rnd --num_exploration_steps=20000 --awac_lambda={l} --iql_expectile={e} --exp_name q5_easy_supervised_lam{l}tau{e}",
"python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 --use_rnd --num_exploration_steps=20000 --unsupervised_exploration --awac_lambda={l} --iql_expectile={e} --exp_name q5_easy_unsupervised_lam{l}tau{e}",
"python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0   --use_rnd --num_exploration_steps=20000 --awac_lambda={l} --iql_expectile={e} --exp_name q5_iql_medium_supervised_lam{l}tau{e}",
"python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --unsupervised_exploration --awac_lambda={l} --iql_expectile={e} --exp_name q5_iql_medium_unsupervised_lam{l}tau{e}",
]

iql_e = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

commands = []
for command in command_stem:
    for e in iql_e:
        commands.append(command.format(e=e))

if __name__ == "__main__":
    # for command in commands:
    #     print(command)
    # user_input = None
    # while user_input not in ['y', 'n']:
    #     user_input = input('Run experiment with above commands? (y/n): ')
    #     user_input = user_input.lower()[:1]
    # if user_input == 'n':
    #     exit(0)
    for command in commands:
        args = shlex.split(command)
        # subprocess.Popen(args)
        subprocess.run(args)