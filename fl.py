from abc import ABC, abstractmethod
import time
import copy
import random
import os
from typing import Any, Dict
import datetime

import numpy as np
from sympy import round_two
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from utils.util_sys import get_available_device, intersection_of_lists
from utils.util_data import get_client_data_loader
from utils.util_data import get_global_test_data_loader
from utils.util_model import get_client_model
from utils.util_model import (
    ipm_attack_craft_model,
    scaling_attack,
    alie_attack,
)
from utils.util_model import get_server_model
from utils.util_fusion import (
    fusion_avg,
    fusion_clipping_median,
    fusion_cos_defense,
    fusion_fedavg,
    fusion_krum,
    fusion_median,
    fusion_trimmed_mean,
    fusion_dendro_defense,
    fusion_dual_defense,
)
from utils.util_logger import logger

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SimulationFL(ABC):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.device = config.get("device", None)

        self.num_clients = config.get("num_clients", 5)
        self.clients_per_round = config.get("clients_per_round", 10)
        self.dataset = config.get("dataset", "mnist")
        self.fusion = config.get("fusion", "fedavg")
        self.partion_type = config.get("partition_type", "noniid")
        self.partion_dirichlet_beta = config.get("partition_dirichlet_beta", 0.25)
        self.dir_data = config.get("dir_data", "./data/")

        self.training_round = config.get("training_round", 10)
        self.local_epochs = config.get("local_epochs", 1)
        self.optimizer = config.get("optimizer", "sgd")
        self.learning_rate = config.get("learning_rate", 0.01)
        self.batch_size = config.get("batch_size", 64)
        self.regularization = config.get("regularization", 1e-5)

        self.attacker_ratio = config.get("attacker_ratio", 0.0)
        self.attacker_strategy = config.get("attacker_strategy", None)
        self.attacker_list = []
        self.attacker_strategy_map = {}  # Maps client_id to attack strategy for multi-attack
        self.attack_start_round = config.get("attack_start_round", -1)
        self.epsilon = config.get("epsilon", None)
        self.data_balance_strategy = config.get("data_balance_strategy", "balanced")
        self.imbalance_percentage = config.get("imbalance_percentage", 0.1)
        self.plot_partition_stats = config.get("plot_partition_stats", False)
        self.min_samples_per_client = config.get("min_samples_per_client", 100)
        self.max_samples_per_client = config.get("max_samples_per_client", 2000)
        
        # 添加IPM攻击的multiplier参数
        self.ipm_multiplier = config.get("ipm_multiplier", 5)
        # 添加ALIE攻击的alie_epsilon参数
        self.alie_epsilon = config.get("alie_epsilon", 0.1)

        # Sample-based scaling attack parameters
        # default to 10 times of the original datasize
        self.fake_data_size_multiplier = config.get("fake_data_size_multiplier", 10.0)
        # 存储恶意客户端伪造的数据大小
        self.fake_data_sizes = {}
        
        # Label flipping attack parameters
        self.label_flipping_ratio = config.get("label_flipping_ratio", 0.5)
        
        # control use FHE or not
        self.use_fhe = config.get("use_fhe", False)

        self.metrics = {}
        self.tensorboard = config.get("tensorboard", None)
        self.log_dir = config.get("log_dir", "logs")
        self.filename_core = config.get(
            "filename_core", f"run_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
        )
        # setup random seed
        self.seed = config.get("seed", 1001)

    def init_seed(self) -> None:
        if self.seed is not None and self.seed > 0:
            logger.info("setting up the seed as {}".format(self.seed))
            set_seed(self.seed)
        else:
            logger.info("no seed is set")

    def init_data(self) -> None:
        self.client_data_loader = get_client_data_loader(
            self.dataset,
            self.dir_data,
            self.num_clients,
            self.partion_type,
            self.partion_dirichlet_beta,
            self.batch_size,
            seed=self.seed,  # Explicitly pass seed as named parameter
            data_balance_strategy=self.data_balance_strategy,
            imbalance_percentage=self.imbalance_percentage,
            plot_partition_stats_flag=self.plot_partition_stats,
            log_dir=os.path.join(self.log_dir, self.filename_core),
            min_samples_per_client=self.min_samples_per_client,
            max_samples_per_client=self.max_samples_per_client,
        )
        self.server_test_data_loader = get_global_test_data_loader(
            self.dataset, self.dir_data, self.batch_size, seed=self.seed
        )

    def init_model(self) -> None:
        self.client_model = get_client_model(
            self.dataset, self.num_clients, self.device
        )
        self.server_model = get_server_model(self.dataset, self.device)

    def init_client_per_round(self) -> None:
        """Initialize the client list for each training round

        This method generates a list of client IDs to participate in each training round
        and stores it in self.round_client_list. The main logic:
        1. Determines number of clients per round (configurable via clients_per_round)
        2. Generates full client ID list based on imbalance setting
        3. Randomly samples clients for each round and sorts them
        4. Uses full client list if all clients participate every round

        Attributes:
            self.round_client_list: 2D list storing client IDs for each round
        """
        num_client_per_round = min(self.num_clients, self.clients_per_round)
        # for extreme imbalanced
        # if self.imbalance:
        #     client_list_all = [i for i in range(1, self.num_clients)]
        # else:
        #     client_list_all = [i for i in range(self.num_clients)]
        
        client_list_all = [i for i in range(self.num_clients)]
        
        round_client_list = []
        if num_client_per_round != self.num_clients:
            for _ in range(self.training_round):
                _client_list = random.sample(client_list_all, num_client_per_round)
                _client_list.sort()
                round_client_list.append(_client_list)
        else:
            for _ in range(self.training_round):
                round_client_list.append(client_list_all)
        self.round_client_list = round_client_list

    def init_attackers(self) -> None:
        """Initialize the attacker list at the beginning

        This method selects attackers from all clients based on the attacker ratio
        and stores them in self.attacker_list. The attackers remain fixed across all rounds.
        For multi-attack strategies, it distributes attackers evenly across different attack types.
        """
        if (self.attacker_strategy is not None 
            and self.attacker_strategy != "none" 
            and self.attacker_ratio > 0):
            num_attackers = int(self.attacker_ratio * self.num_clients)
            # Ensure at least one attacker if ratio > 0
            num_attackers = max(1, num_attackers)
            
            # Select all attackers first
            all_attackers = random.sample(list(range(self.num_clients)), num_attackers)
            self.attacker_list = all_attackers
            
            # For multi-attack strategies, distribute attackers across different attack types
            if self.attacker_strategy.startswith("multi_attack"):
                # Initialize a dictionary to store attack type for each attacker
                self.attacker_strategy_map = {}
                
                # Determine attack types from the strategy name
                attack_types = []
                if self.attacker_strategy == "multi_attack_ipm+scaling":
                    attack_types = ["model_poisoning_ipm", "model_poisoning_scaling"]
                elif self.attacker_strategy == "multi_attack_ipm+alie":
                    attack_types = ["model_poisoning_ipm", "model_poisoning_alie"]
                elif self.attacker_strategy == "multi_attack_scaling+alie":
                    attack_types = ["model_poisoning_scaling", "model_poisoning_alie"]
                elif self.attacker_strategy == "multi_attack_all":
                    attack_types = ["model_poisoning_ipm", "model_poisoning_scaling", "model_poisoning_alie"]
                
                # Calculate how many attackers to assign to each attack type
                num_attack_types = len(attack_types)
                base_num_per_type = num_attackers // num_attack_types
                remainder = num_attackers % num_attack_types
                
                # Distribute attackers evenly across attack types
                attacker_idx = 0
                for i, attack_type in enumerate(attack_types):
                    # Assign additional attacker from remainder if needed
                    count_for_this_type = base_num_per_type + (1 if i < remainder else 0)
                    
                    # Assign attack type to each attacker
                    for j in range(count_for_this_type):
                        if attacker_idx < len(all_attackers):
                            self.attacker_strategy_map[all_attackers[attacker_idx]] = attack_type
                            attacker_idx += 1
                
                # Log the distribution
                attack_distribution = {}
                for attack_type in attack_types:
                    count = sum(1 for t in self.attacker_strategy_map.values() if t == attack_type)
                    attack_distribution[attack_type] = count
                
                logger.info(f"Selected attackers: {self.attacker_list} ({len(self.attacker_list)}/{self.num_clients})")
                logger.info(f"Attack distribution: {attack_distribution}")
            else:
                # For single attack strategy, all attackers use the same strategy
                logger.info(f"Selected attackers: {self.attacker_list} ({len(self.attacker_list)}/{self.num_clients})")
        else:
            self.attacker_list = []
            logger.info("No attackers selected")

    def init_device(self) -> None:
        self.device = get_available_device()

    def start(self):
        self.init_device()
        self.init_seed()
        self.init_client_per_round()
        self.init_attackers()  # 初始化攻击者列表
        self.init_data()
        self.init_model()

        logger.info("start the FL simulation")

        time_start = time.perf_counter()

        for _round_idx in range(self.training_round):
            logger.info(f"start training round {_round_idx}")
            logger.info(f"participating clients in this round: {self.round_client_list[_round_idx]}")
            self.metrics[_round_idx] = {"time": None, "parties": {}, "server": {}}

            # simulate query each client
            server_model_params = self.server_model.state_dict()
            round_client_list = self.round_client_list[_round_idx]
            round_client_models = {
                pid: self.client_model[pid] for pid in round_client_list
            }

            for _pid, _model in round_client_models.items():
                _model.load_state_dict(server_model_params)

            # 检查本轮参与的客户端中哪些是攻击者
            attackers_this_round = intersection_of_lists(round_client_list, self.attacker_list)
            if (
                self.attacker_strategy is not None
                and self.attacker_strategy != "none"
                and len(attackers_this_round) > 0
                and _round_idx >= self.attack_start_round
            ):
                # 对于混合攻击策略，按照攻击类型分组显示攻击者
                if self.attacker_strategy.startswith("multi_attack") and hasattr(self, 'attacker_strategy_map'):
                    attack_type_to_clients = {}
                    for client_id in attackers_this_round:
                        attack_type = self.attacker_strategy_map.get(client_id, "unknown")
                        if attack_type not in attack_type_to_clients:
                            attack_type_to_clients[attack_type] = []
                        attack_type_to_clients[attack_type].append(client_id)
                    
                    for attack_type, clients in attack_type_to_clients.items():
                        logger.info(f"round {_round_idx} attackers using {attack_type}: {clients}")
                else:
                    logger.info(f"round {_round_idx} attackers: {attackers_this_round}")
            else:
                logger.info(f"no attack at the round {_round_idx}")

            # simulate local training in parallel
            model_dict = {}
            for _client_id in round_client_list:
                logger.info(f"start client {_client_id} training")
                model_client, eval_metrics = self.client_local_train(
                    _round_idx, _client_id, round_client_models[_client_id]
                )
                model_dict[_client_id] = model_client
                logger.info(f"end client {_client_id} training")

                # RECORD PARTY METRICS
                logger.info(f"client {_client_id} evaluation metrics: {eval_metrics}")
                self.metrics[_round_idx]["parties"][_client_id] = eval_metrics

            # simulate aggregation
            aggregated_params = self.aggregate_model(_round_idx, model_dict)
            self.server_model.load_state_dict(aggregated_params)

            # RECORD GLOBAL METRICS
            criterion = nn.CrossEntropyLoss().to(self.device)
            _, _test_acc = self.model_evaluate(
                self.server_model, self.server_test_data_loader, criterion
            )
            self.metrics[_round_idx]["server"]["test_acc"] = _test_acc
            logger.info(f"global side -  test accuracy: {_test_acc}")
            self.tensorboard.add_scalar(
                "{}-{} - Server Test Acc".format(
                    self.dataset,
                    self.fusion,
                ),
                _test_acc,
                _round_idx,
            )
            
            # 计算每个类别的测试准确率
            per_class_accuracy = self.evaluate_per_class_accuracy(
                self.server_model, self.server_test_data_loader
            )
            
            # 记录每个类别的准确率
            logger.info(f"Per-class accuracy in round {_round_idx}:")
            for class_id, acc in per_class_accuracy.items():
                logger.info(f"  Class {class_id}: {acc:.2f}%")
                self.metrics[_round_idx]["server"][f"class_{class_id}_acc"] = acc
                self.tensorboard.add_scalar(
                    f"{self.dataset}-{self.fusion} - Class {class_id} Acc",
                    acc,
                    _round_idx,
                )
            
            self.tensorboard.flush()

            time_round_end = time.perf_counter()
            self.metrics[_round_idx]["time"] = time_round_end - time_start

        logger.info("end the FL simulation")
        logger.info("summarization - simulation metrics: {}".format(self.metrics))
        self.tensorboard.close()

    def client_local_train(
        self, round_idx: int, client_id: int, client_model: nn.Module
    ) -> None:

        logger.info(f"client {client_id} start local training ...")
        train_data_loader, test_data_loader = self.client_data_loader[client_id]
        
        set_seed(round_idx * self.num_clients + client_id)

        model = client_model.to(self.device)
        model.train()
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = self.get_optimizer(model)

        # 检查是否是执行label flipping攻击的客户端
        is_label_flipping_attacker = (
            client_id in self.attacker_list
            and round_idx >= self.attack_start_round
            and (
                self.attacker_strategy == "model_poisoning_scaling" or
                (hasattr(self, 'attacker_strategy_map') and 
                 client_id in self.attacker_strategy_map and 
                 self.attacker_strategy_map[client_id] == "model_poisoning_scaling")
            )
        )

        for _epoch in range(self.local_epochs):
            train_loss_lst = []
            epoch_correct = 0
            epoch_total = 0

            _size_total_data = len(train_data_loader.dataset)
            _size_batch = len(train_data_loader)

            for _batch_idx, (_data, _target) in enumerate(train_data_loader):
                data, target = _data.to(self.device), _target.to(self.device)
                
                # 如果是执行label flipping的攻击者，执行label flipping攻击
                if is_label_flipping_attacker:
                    # 随机选择一定比例的样本进行标签翻转
                    batch_size = target.size(0)
                    flip_indices = torch.rand(batch_size, device=self.device) < self.label_flipping_ratio
                    
                    if flip_indices.any():
                        # 获取数据集的类别数量
                        if self.dataset == "mnist" or self.dataset == "fmnist":
                            num_classes = 10
                        elif self.dataset == "cifar10" or self.dataset == "svhn":
                            num_classes = 10
                        elif self.dataset == "emnist_byclass":
                            num_classes = 62
                        elif self.dataset == "emnist_bymerge":
                            num_classes = 47
                        elif self.dataset == "tinyimagenet":
                            num_classes = 200
                        else:
                            num_classes = 10  # 默认值
                        
                        # 对选中的样本进行标签翻转
                        # 将标签随机翻转为另一个类别
                        flipped_labels = torch.randint(0, num_classes - 1, size=(flip_indices.sum(),), device=self.device)
                        # 确保翻转后的标签与原标签不同
                        for i, orig_label in enumerate(target[flip_indices]):
                            if flipped_labels[i] >= orig_label:
                                flipped_labels[i] += 1
                        
                        # 应用翻转的标签
                        target_copy = target.clone()
                        target_copy[flip_indices] = flipped_labels
                        target = target_copy

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss_lst.append(loss.item())
                _, predicted = torch.max(output.data, 1)
                epoch_total += target.size(0)
                epoch_correct += (predicted == target.data).sum().item()

            epoch_train_acc = epoch_correct / epoch_total * 100
            epoch_avg_loss = np.mean(train_loss_lst)

        if (
            client_id in self.attacker_list
            and round_idx >= self.attack_start_round
        ):
            # Determine the attack strategy for this client
            attack_strategy = self.attacker_strategy
            if hasattr(self, 'attacker_strategy_map') and client_id in self.attacker_strategy_map:
                attack_strategy = self.attacker_strategy_map[client_id]
            
            # Execute the appropriate attack based on the strategy
            if attack_strategy == "model_poisoning_ipm":
                logger.info(f"client {client_id} is attacker, start poisoning model with IPM attack")
                crafted_model = ipm_attack_craft_model(
                    self.server_model.to(self.device), model.to(self.device), 
                    multiplier=self.ipm_multiplier
                )
                _, _test_acc = self.model_evaluate(
                    crafted_model, test_data_loader, criterion
                )
                _train_loss, _train_acc = self.model_evaluate(
                    crafted_model, train_data_loader, criterion
                )
                return crafted_model, {
                    "train_loss": _train_loss,
                    "train_acc": _train_acc,
                    "test_acc": _test_acc,
                }
            elif attack_strategy == "model_poisoning_scaling":
                logger.info(f"client {client_id} is attacker, start sample-based scaling attack")
                # 不修改模型参数，只返回正常训练后的模型
                # 实际的攻击在aggregate_model中通过谎报数据大小实现
                
                # 记录真实的数据大小，以便在aggregate_model中使用
                real_data_size = len(train_data_loader.dataset)
                # 计算伪造的数据大小（默认为真实大小的10倍）
                fake_data_size = int(real_data_size * self.fake_data_size_multiplier)
                # 存储伪造的数据大小，以便在aggregate_model中使用
                self.fake_data_sizes[client_id] = fake_data_size
                
                logger.info(f"Client {client_id} real data size: {real_data_size}, fake data size: {fake_data_size}")
                
                _test_loss, _test_acc = self.model_evaluate(
                    model, test_data_loader, criterion
                )
                _train_loss, _train_acc = self.model_evaluate(
                    model, train_data_loader, criterion
                )
                return model, {
                    "train_loss": _train_loss,
                    "train_acc": _train_acc,
                    "test_acc": _test_acc,
                }
            elif attack_strategy == "model_poisoning_alie":
                logger.info(f"client {client_id} is attacker, start poisoning model with ALIE attack")
                crafted_model = alie_attack(model.to(self.device), epsilon=self.alie_epsilon)
                _, _test_acc = self.model_evaluate(
                    crafted_model, test_data_loader, criterion
                )
                _train_loss, _train_acc = self.model_evaluate(
                    crafted_model, train_data_loader, criterion
                )
                return crafted_model, {
                    "train_loss": _train_loss,
                    "train_acc": _train_acc,
                    "test_acc": _test_acc,
                }
        else:
            # Normal client behavior
            _test_loss, _test_acc = self.model_evaluate(
                model, test_data_loader, criterion
            )
            _train_loss, _train_acc = self.model_evaluate(
                model, train_data_loader, criterion
            )
            return model, {
                "train_loss": _train_loss,
                "train_acc": _train_acc,
                "test_acc": _test_acc,
            }

    def _batch_records_debug(
        self,
        epoch: int,
        batch_idx: int,
        size_total_data: int,
        size_data: int,
        size_batch: int,
        loss: Any,
    ) -> None:
        if batch_idx % 10 == 0:
            logger.debug(
                "train epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}".format(
                    epoch,
                    batch_idx * size_data,
                    size_total_data,
                    100.0 * batch_idx / size_batch,
                    loss.item(),
                )
            )

    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        if self.optimizer == "adam":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.learning_rate,
                weight_decay=self.regularization,
            )
        elif self.optimizer == "amsgrad":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.learning_rate,
                weight_decay=self.regularization,
                amsgrad=True,
            )
        elif self.optimizer == "sgd":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.regularization,
            )
        return optimizer

    def model_evaluate(
        self,
        model: nn.Module,
        data_loader: data.DataLoader,
        criterion: nn.CrossEntropyLoss,
    ) -> tuple:
        model.eval()

        loss = 0
        correct = 0
        with torch.no_grad():
            model.to(self.device)
            for _data, _targets in data_loader:
                data, targets = _data.to(self.device), _targets.to(self.device)
                outputs = model(data)
                loss += criterion(outputs, targets).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == targets).sum().item()

        loss /= len(data_loader)
        accuracy = 100.0 * correct / len(data_loader.dataset)
        return loss, accuracy
        
    def _log_attack_type_metrics(self, round_idx, attackers_this_round, selected_benigns):
        """Log metrics broken down by attack type for multi-attack scenarios"""
        if not self.attacker_strategy.startswith("multi_attack") or not hasattr(self, 'attacker_strategy_map'):
            return
            
        # Find attackers that were falsely identified as benign
        false_negative_clients = attackers_this_round & selected_benigns
        if not false_negative_clients:
            return
            
        fn_by_type = {}
        for client_id in false_negative_clients:
            attack_type = self.attacker_strategy_map.get(client_id, "unknown")
            fn_by_type[attack_type] = fn_by_type.get(attack_type, 0) + 1
        
        # Calculate FN rate by attack type
        fn_rate_by_type = {}
        for attack_type, fn_count in fn_by_type.items():
            # Count total attackers of this type
            total_of_type = sum(1 for cid in attackers_this_round 
                              if self.attacker_strategy_map.get(cid, "") == attack_type)
            fn_rate_by_type[attack_type] = fn_count / total_of_type if total_of_type > 0 else 0.0
        
        logger.info(f"[Metrics] FN breakdown by attack type: {fn_by_type}")
        logger.info(f"[Metrics] FN rate by attack type: {fn_rate_by_type}")
        
        # Log to TensorBoard
        if self.tensorboard is not None:
            for attack_type, rate in fn_rate_by_type.items():
                self.tensorboard.add_scalar(f"DualDefense/FN_rate_{attack_type}", rate, round_idx)

    def evaluate_per_class_accuracy(
        self,
        model: nn.Module,
        data_loader: data.DataLoader,
    ) -> Dict[int, float]:
        """
        计算每个类别的测试准确率
        
        Args:
            model: 要评估的模型
            data_loader: 测试数据加载器
            
        Returns:
            每个类别的准确率字典 {class_id: accuracy}
        """
        model.eval()
        
        # 获取数据集中的类别数量
        if self.dataset == "mnist" or self.dataset == "fmnist":
            num_classes = 10
        elif self.dataset == "cifar10":
            num_classes = 10
        elif self.dataset == "svhn":
            num_classes = 10
        elif self.dataset == "emnist_byclass":
            num_classes = 62
        elif self.dataset == "emnist_bymerge":
            num_classes = 47
        elif self.dataset == "tinyimagenet":
            num_classes = 200
        else:
            num_classes = 10  # 默认为10个类别
            
        # 初始化每个类别的正确预测数和总样本数
        class_correct = {i: 0 for i in range(num_classes)}
        class_total = {i: 0 for i in range(num_classes)}
        
        with torch.no_grad():
            model.to(self.device)
            for _data, _targets in data_loader:
                data, targets = _data.to(self.device), _targets.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                
                # 更新每个类别的统计信息
                for i in range(len(targets)):
                    label = targets[i].item()
                    class_total[label] += 1
                    if predicted[i] == targets[i]:
                        class_correct[label] += 1
        
        # 计算每个类别的准确率
        class_accuracy = {}
        for i in range(num_classes):
            if class_total[i] > 0:
                class_accuracy[i] = 100.0 * class_correct[i] / class_total[i]
            else:
                class_accuracy[i] = 0.0
                
        return class_accuracy

    def aggregate_model(self, round_idx: int, model_updates: dict) -> Dict[str, Any]:
        logger.info("start model aggregation...fusion method: {}".format(self.fusion))

        # 获取真实的数据大小
        real_data_sizes = {
            p_id: sum(len(batch[0]) for batch in self.client_data_loader[p_id][0])
            for p_id in self.round_client_list[round_idx]
        }
        
        # 创建一个新的数据大小字典，对于攻击者使用伪造的数据大小
        data_sizes = real_data_sizes.copy()
        
        # 检查是否有攻击者
        attackers_in_round = [client_id for client_id in self.round_client_list[round_idx] 
                            if client_id in self.attacker_list]
        
        # 只有在达到攻击开始轮次后，才执行攻击行为
        if round_idx >= self.attack_start_round:
            # 对于每个攻击者，根据其攻击策略执行相应的操作
            for attacker_id in attackers_in_round:
                # 确定攻击者的攻击策略
                attack_strategy = self.attacker_strategy
                if hasattr(self, 'attacker_strategy_map') and attacker_id in self.attacker_strategy_map:
                    attack_strategy = self.attacker_strategy_map[attacker_id]
                
                # 如果是scaling攻击，使用伪造的数据大小
                if attack_strategy == "model_poisoning_scaling" and attacker_id in self.fake_data_sizes:
                    fake_size = self.fake_data_sizes[attacker_id]
                    real_size = real_data_sizes[attacker_id]
                    data_sizes[attacker_id] = fake_size
                    logger.info(f"Attacker {attacker_id} (using {attack_strategy}) reporting fake data size: {fake_size} (real: {real_size})")
            
            # 记录不同攻击类型的统计信息
            if hasattr(self, 'attacker_strategy_map') and attackers_in_round:
                attack_type_counts = {}
                for attacker_id in attackers_in_round:
                    attack_type = self.attacker_strategy_map.get(attacker_id, self.attacker_strategy)
                    attack_type_counts[attack_type] = attack_type_counts.get(attack_type, 0) + 1
                logger.info(f"Attackers by type in round {round_idx}: {attack_type_counts}")
            
            # 计算攻击者伪造数据大小占总数据大小的比例
            total_data_size = sum(data_sizes.values())
            attackers_weight = sum(data_sizes[a_id] for a_id in attackers_in_round) / total_data_size if total_data_size > 0 else 0
            logger.info(f"Attackers' weight in aggregation: {attackers_weight:.4f} ({len(attackers_in_round)} attackers)")
        else:
            # 在攻击开始轮次之前，记录潜在的攻击者，但不执行攻击行为
            if attackers_in_round:
                logger.info(f"Round {round_idx} before attack_start_round ({self.attack_start_round}): {len(attackers_in_round)} potential attackers not active yet")
        
        logger.debug("data sizes for aggregation: {}".format(data_sizes))

        if self.fusion == "average":
            average_params = fusion_avg(model_updates)
            return average_params
        elif self.fusion == "fedavg":
            weighted_avg_params = fusion_fedavg(model_updates, data_sizes)
            return weighted_avg_params
        elif self.fusion == "krum":
            # max_expected_adversaries = int(self.attacker_ratio * self.num_clients)
            max_expected_adversaries = int(self.attacker_ratio * len(model_updates))
            krum_params = fusion_krum(
                model_updates, max_expected_adversaries, self.device
            )
            return krum_params
        elif self.fusion == "median":
            median_params = fusion_median(model_updates, device=self.device)
            return median_params
        elif self.fusion == "clipping_median":
            median_clipping_params = fusion_clipping_median(
                model_updates, clipping_threshold=0.1, device=self.device
            )
            return median_clipping_params
        elif self.fusion == "trimmed_mean":
            trimmed_mean_params = fusion_trimmed_mean(
                model_updates, trimmed_ratio=0.1, device=self.device
            )
            return trimmed_mean_params
        elif self.fusion == "cos_defense":
            # 对于cos_defense，我们也使用伪造的数据大小
            weighted_params = fusion_cos_defense(self.server_model, model_updates, data_sizes)
            return weighted_params
        elif self.fusion == "dual_defense":
            logger.info("start dual-defense fusion")
            lst_round_attackers = intersection_of_lists(
                list(model_updates.keys()), self.attacker_list
            )
            logger.info(f"round {round_idx} attackers: {lst_round_attackers}")
            
            # 对于dual_defense，使用原始算法进行聚合
            fused_params, benigns = fusion_dual_defense(
                self.server_model,
                model_updates,
                data_sizes,
                _round_idx=round_idx,
                log_dir=os.path.join(self.log_dir, self.filename_core),
                attacker_list=self.attacker_list,
                use_fhe=self.use_fhe,
                gradient_collector=None,
                similarity_threshold=None,
                epsilon=None,
            )
            
            # ---------- calculate FN / FP rate ----------
            # 获取当前轮次参与的客户端
            all_clients = set(model_updates.keys())
            selected_benigns = set(benigns)
            
            # 在攻击开始轮次之前，不应该计算FN（因为没有真正的攻击者）
            if round_idx >= self.attack_start_round:
                # 获取当前轮次参与的攻击者
                attackers_this_round = all_clients & set(self.attacker_list)
                true_benigns = all_clients - attackers_this_round
                
                # 假阴性：被错误判定为良性的攻击者
                fn = len(attackers_this_round & selected_benigns)
                # 假阳性：被错误判定为恶意的良性客户端
                fp = len(true_benigns - selected_benigns)
                
                # 使用当前轮次参与的攻击者数量作为分母
                fn_rate = fn / len(attackers_this_round) if attackers_this_round else 0.0
                fp_rate = fp / len(true_benigns) if true_benigns else 0.0
                
                logger.info(f"[Metrics] Round {round_idx}  FN={fn} ({fn_rate:.2%})  FP={fp} ({fp_rate:.2%})")
            else:
                # 在攻击开始轮次之前，所有客户端都是良性的
                true_benigns = all_clients
                
                # 假阴性始终为0（因为没有真正的攻击者）
                fn = 0
                fn_rate = 0.0
                
                # 假阳性：被错误判定为恶意的良性客户端
                fp = len(true_benigns - selected_benigns)
                fp_rate = fp / len(true_benigns) if true_benigns else 0.0
                
                logger.info(f"[Metrics] Round {round_idx} (before attack start)  FN={fn} (0.00%)  FP={fp} ({fp_rate:.2%})")

            # TensorBoard
            if self.tensorboard is not None:
                self.tensorboard.add_scalar("DualDefense/FN_rate", fn_rate, round_idx)
                self.tensorboard.add_scalar("DualDefense/FP_rate", fp_rate, round_idx)
            
            return fused_params
            
        elif self.fusion == "dendro_defense":
            logger.info("start dendro-defense fusion")
            lst_round_attackers = intersection_of_lists(
                list(model_updates.keys()), self.attacker_list
            )
            logger.info(f"round {round_idx} attackers: {lst_round_attackers}")
            
            # 对于dendro_defense，我们也使用伪造的数据大小
            # 这样可以测试防御机制是否能够检测到异常
            fused_params, benigns = fusion_dendro_defense(
                self.server_model,
                model_updates,
                data_sizes,
                _round_idx=round_idx,
                log_dir=os.path.join(self.log_dir, self.filename_core),
                attacker_list=self.attacker_list,
                use_fhe=self.use_fhe,
                gradient_collector=None,
            )
            # ---------- calculate FN / FP rate ----------
            # 获取当前轮次参与的客户端
            all_clients = set(model_updates.keys())
            selected_benigns = set(benigns)
            
            # 在攻击开始轮次之前，不应该计算FN（因为没有真正的攻击者）
            if round_idx >= self.attack_start_round:
                # 获取当前轮次参与的攻击者
                attackers_this_round = all_clients & set(self.attacker_list)
                true_benigns = all_clients - attackers_this_round
                
                # 假阴性：被错误判定为良性的攻击者
                fn = len(attackers_this_round & selected_benigns)
                # 假阳性：被错误判定为恶意的良性客户端
                fp = len(true_benigns - selected_benigns)
                
                # 使用当前轮次参与的攻击者数量作为分母
                fn_rate = fn / len(attackers_this_round) if attackers_this_round else 0.0
                fp_rate = fp / len(true_benigns) if true_benigns else 0.0
                
                logger.info(f"[Metrics] Round {round_idx}  FN={fn} ({fn_rate:.2%})  FP={fp} ({fp_rate:.2%})")
            else:
                # 在攻击开始轮次之前，所有客户端都是良性的
                true_benigns = all_clients
                
                # 假阴性始终为0（因为没有真正的攻击者）
                fn = 0
                fn_rate = 0.0
                
                # 假阳性：被错误判定为恶意的良性客户端
                fp = len(true_benigns - selected_benigns)
                fp_rate = fp / len(true_benigns) if true_benigns else 0.0
                
                logger.info(f"[Metrics] Round {round_idx} (before attack start)  FN={fn} (0.00%)  FP={fp} ({fp_rate:.2%})")

            # TensorBoard
            if self.tensorboard is not None:
                self.tensorboard.add_scalar("DualDefense/FN_rate", fn_rate, round_idx)
                self.tensorboard.add_scalar("DualDefense/FP_rate", fp_rate, round_idx)
            
            return fused_params
        else:
            raise ValueError("Invalid fusion method")
