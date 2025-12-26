from __future__ import print_function

import argparse


def args_parser():
    parser = argparse.ArgumentParser(
        description="a simulation of federated learning with defense mechanisms."
    )

    # federated settings
    parser.add_argument(
        "-np",
        "--num_clients",
        type=int,
        default=100,
        help="number of workers in a distributed cluster",
    )
    parser.add_argument(
        "-cpr",
        "--clients_per_round",
        type=int,
        default=10,
        help="number of clients participating in each training round",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fmnist", "cifar10", "svhn", "emnist_byclass", "emnist_bymerge", "tinyimagenet"],
        help="dataset used for training",
    )
    parser.add_argument(
        "-pd",
        "--partition_type",
        type=str,
        default="iid",
        choices=["noniid", "iid", "dirichlet_fixed", "iid_quantity_skew"],
        help="data partitioning strategy",
    )
    parser.add_argument(
        "-pb",
        "--partition_dirichlet_beta",
        type=float,
        default=0.5,
        help="dirichlet distribution parameter for data partitioning",
    )
    parser.add_argument(
        "-f",
        "--fusion",
        choices=[
            "average",
            "fedavg",
            "krum",
            "median",
            "clipping_median",
            "trimmed_mean",
            "cos_defense",
            "dendro_defense",
            "dual_defense",
        ],
        type=str,
        default="dendro_defense",
        help="dirichlet distribution parameter for data partitioning",
    )
    parser.add_argument(
        "-dm",
        "--dir_model",
        type=str,
        required=False,
        default="./models/",
        help="Model directory path",
    )
    parser.add_argument(
        "-dd",
        "--dir_data",
        type=str,
        required=False,
        default="./data/",
        help="Data directory",
    )

    # hyperparameters settings
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.005,
        help="learning rate (default: 0.005)",
    )
    parser.add_argument(
        "-le", "--local_epochs", type=int, default=5, help="number of local epochs"
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=64,
        help="input batch size for training",
    )
    parser.add_argument(
        "-tr",
        "--training_round",
        type=int,
        default=100,
        help="number of maximum communication roun",
    )
    parser.add_argument(
        "-re",
        "--regularization",
        type=float,
        default=1e-5,
        help="L2 regularization strength",
    )
    parser.add_argument(
        "-op",
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam", "amsgrad"],
        help="optimizer used in the training process",
    )
    # adv and defense settings
    parser.add_argument(
        "--attacker_ratio",
        type=float,
        default=0.3,
        required=False,
        help="ratio for number of attackers",
    )
    parser.add_argument(
        "--attacker_strategy",
        type=str,
        default="none",
        required=False,
        choices=[
            "none",
            "model_poisoning_ipm",
            "model_poisoning_scaling",
            "model_poisoning_alie",
            "data_poisoning_label_flip",
            "multi_attack_ipm+scaling",
            "multi_attack_ipm+alie",
            "multi_attack_scaling+alie",
            "multi_attack_all",
        ],
        help="attacker strategy, supports single or multiple attack types",
    )
    parser.add_argument(
        "--attack_start_round",
        type=int,
        default=50,
        required=False,
        help="the round to start attack",
    )
    parser.add_argument(
        "--label_flipping_ratio",
        type=float,
        default=0.5,
        required=False,
        help="ratio of training data to flip labels for malicious clients",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        required=False,
        help="[DEPRECATED] No longer used by dendro_defense method",
    )
    # 添加IPM攻击的multiplier参数
    parser.add_argument(
        "--ipm_multiplier",
        type=int,
        default=5,
        required=False,
        help="multiplier parameter for IPM attack (previously called 'action')",
    )
    # 添加ALIE攻击的alie_epsilon参数
    parser.add_argument(
        "--alie_epsilon",
        type=float,
        default=0.1,
        required=False,
        help="epsilon parameter for ALIE attack",
    )
    # parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label')
    # parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Trigger Path')
    # parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size')

    # other settings
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        required=False,
        choices=["cpu", "mps", "cuda"],
        help="device to run the program with pytorch",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7890,
        required=False,
        help="Random seed",
    )
    parser.add_argument(
        "-ld",
        "--log_dir",
        type=str,
        default="./logs/",
        help="Directory for storing logs and TensorBoard files.",
    )
    parser.add_argument(
        "--data_balance_strategy",
        type=str,
        default="balanced",
        choices=["balanced", "extreme_imbalance", "soft_imbalance"],
        help="Strategy for data balancing across clients. "
        "'balanced': default behavior. "
        "'extreme_imbalance': client 0 gets a small portion of class 0, other clients get other classes. "
        "'soft_imbalance': remove most of class 0 data, then distribute.",
    )
    parser.add_argument(
        "--imbalance_percentage",
        type=float,
        default=0.1,
        help="Percentage of class 0 samples to keep for client 0 in imbalanced mode (default: 0.1)",
    )
    parser.add_argument(
        "--disable_fhe",
        type=int,
        default=1,
        required=False,
        help="Disable Homomorphic Encryption in the DualDefense mechanism (0: enable FHE, 1: disable FHE)",
    )
    parser.add_argument(
        "--plot_partition_stats",
        action="store_true",
        help="Plot and save client data distribution stats.",
        default=True,
    )
    parser.add_argument(
        "--min_samples_per_client",
        type=int,
        default=100,
        help="Minimum number of samples per client for 'iid_quantity_skew'.",
    )
    parser.add_argument(
        "--max_samples_per_client",
        type=int,
        default=2000,
        help="Maximum number of samples per client for 'iid_quantity_skew'.",
    )
    return parser.parse_args()
