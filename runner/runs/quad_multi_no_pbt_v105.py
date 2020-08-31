from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [0000, 1111, 2222]),
    ('batch_size', [512]),
    ('ppo_epochs', [1]),
    ('nonlinearity', ['tanh']),
    ('learning_rate', [1e-4]),
    ('exploration_loss_coeff', [0.0]),
    ('initial_stddev', [1.0]),
    ('hidden_size', [64]),
    ('adam_eps', [1e-8]),
])

_experiment = Experiment(
    'quads_gridsearch',
    'python -m run_algorithm --env=quadrotor_multi --train_for_env_steps=1000000000 --algo=APPO --gamma=0.99 --use_rnn=False --num_workers=72 --num_envs_per_worker=2 --num_policies=1 --rollout=128 --recurrence=1 --benchmark=False --with_pbt=False --actor_critic_share_weights=False --policy_initialization=xavier_uniform --adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --gae_lambda=1.00 --max_grad_norm=0.0 --ppo_clip_value=5.0 --quads_episode_duration=8 --quads_num_agents=5',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_multi_v105_3seeds_circ', experiments=[_experiment])
