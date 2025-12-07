import os
import datetime

from args import args_parser

if __name__ == "__main__":
    args = args_parser()
    # print(f"args: {args}")

    # 处理FHE参数
    use_fhe = args.disable_fhe == 0

    # path for log and tensorboard
    log_dir = tensorboard_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    config = {
        "num_clients": args.num_clients,
        "clients_per_round": args.clients_per_round,
        "dataset": args.dataset,
        "fusion": args.fusion,
        "training_round": args.training_round,
        "local_epochs": args.local_epochs,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "data_dir": args.dir_data,
        "partition_type": args.partition_type,
        "partition_dirichlet_beta": args.partition_dirichlet_beta,
        "regularization": args.regularization,
        "attacker_ratio": args.attacker_ratio,
        "attacker_strategy": args.attacker_strategy,
        "attack_start_round": args.attack_start_round,
        "label_flipping_ratio": args.label_flipping_ratio,
        "epsilon": args.epsilon,
        "ipm_multiplier": args.ipm_multiplier,
        "alie_epsilon": args.alie_epsilon,
        "device": args.device,
        "seed": args.seed,
        "data_balance_strategy": args.data_balance_strategy,
        "imbalance_percentage": args.imbalance_percentage,
        "use_fhe": use_fhe,
        "log_dir": log_dir,
        "plot_partition_stats": args.plot_partition_stats,
        "min_samples_per_client": args.min_samples_per_client,
        "max_samples_per_client": args.max_samples_per_client,
    }
    # print(f"config: {config}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    filename_core = "{}-{}-{}-p{}cpr{}r{}e{}b{}ar{}as{}-pb{}-opt{}-reg{}-seed{}-eps{}-{}-{}-fhe{}-lfr{}-ipm{}-alie{}-{}".format(
        args.dataset,
        args.fusion,
        args.attacker_strategy,
        args.num_clients,
        args.clients_per_round,
        args.training_round,
        args.local_epochs,
        args.batch_size,
        args.attacker_ratio,
        args.attack_start_round,
        args.partition_dirichlet_beta,
        args.optimizer,
        args.regularization,
        args.seed,
        args.epsilon,
        args.data_balance_strategy,
        args.partition_type,
        use_fhe,
        args.label_flipping_ratio,
        args.ipm_multiplier,
        args.alie_epsilon,
        timestamp,
    )

    log_file = filename_core + ".log"
    tb_file = filename_core
    os.environ["LOG_FILE_NAME"] = os.path.join(log_dir, log_file)

    from utils.util_logger import setup_tensorboard

    config["tensorboard"] = setup_tensorboard(tensorboard_dir, tb_file)
    config["filename_core"] = filename_core

    from fl import SimulationFL

    fl = SimulationFL(config)
    fl.start()
