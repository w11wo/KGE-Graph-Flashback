import logging
import gflags
import sys
import os
import json
from tqdm import tqdm

tqdm.monitor_iterval = 0
import math
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.utils.data import Dataset, DataLoader

from jTransUP.models.base import get_flags, flag_defaults, init_model
from jTransUP.data.load_triple_data import load_data
from jTransUP.utils.trainer import ModelTrainer
from jTransUP.utils.misc import to_gpu, evalKGProcess, USE_CUDA
from jTransUP.utils.loss import bprLoss, orthogonalLoss, normLoss
from jTransUP.utils.visuliazer import Visualizer
from jTransUP.utils.data import getTrainTripleBatch
import jTransUP.utils.loss as loss

FLAGS = gflags.FLAGS


class KGDataset(Dataset):
    def __init__(self, train_list, entity_total, head_dict, tail_dict):
        self.train_list = train_list
        self.entity_total = entity_total
        self.head_dict = head_dict
        self.tail_dict = tail_dict

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        # Move the negative sampling logic here
        h, t, r = self.train_list[idx]

        # Negative Head
        nh = np.random.randint(0, self.entity_total - 1)
        while nh in self.head_dict.get((t, r), set()):
            nh = np.random.randint(0, self.entity_total - 1)

        # Negative Tail
        nt = np.random.randint(0, self.entity_total - 1)
        while nt in self.tail_dict.get((h, r), set()):
            nt = np.random.randint(0, self.entity_total - 1)

        return torch.LongTensor([h, t, r, nh, nt, r])  # nr is usually same as r


def evaluate(
    FLAGS,
    model,
    entity_total,
    relation_total,
    eval_head_iter,
    eval_tail_iter,
    eval_head_dict,
    eval_tail_dict,
    all_head_dicts,
    all_tail_dicts,
    logger,
    eval_descending=True,
    is_report=False,
):
    # Evaluate
    total_batches = len(eval_head_iter) + len(eval_tail_iter)
    # processing bar
    pbar = tqdm(total=total_batches)
    pbar.set_description("Run Eval")

    model.eval()
    model.disable_grad()

    # head prediction evaluation
    head_results = []
    for batch_trs in eval_head_iter:  # batch_trs: [(t, r), ...]
        t = [tr[0] for tr in batch_trs]
        r = [tr[1] for tr in batch_trs]
        t_var = to_gpu(V(torch.LongTensor(t)))
        r_var = to_gpu(V(torch.LongTensor(r)))
        # batch * item
        scores = model.evaluateHead(t_var, r_var)
        preds = zip(batch_trs, scores.data.cpu().numpy())  # [(t, r): {所有实体h的分数}, ...]

        head_results.extend(  # eval_descending=False,从小到大排序, top k在前面
            evalKGProcess(
                list(preds),
                eval_head_dict,
                all_dicts=all_head_dicts,
                descending=eval_descending,
                num_processes=FLAGS.num_processes,
                topn=FLAGS.topn,
                queue_limit=FLAGS.max_queue,
            )
        )

        pbar.update(1)
    # tail prediction evaluation
    tail_results = []
    for batch_hrs in eval_tail_iter:
        h = [hr[0] for hr in batch_hrs]
        r = [hr[1] for hr in batch_hrs]
        h_var = to_gpu(V(torch.LongTensor(h)))
        r_var = to_gpu(V(torch.LongTensor(r)))
        # batch * item
        scores = model.evaluateTail(h_var, r_var)
        preds = zip(batch_hrs, scores.data.cpu().numpy())

        tail_results.extend(
            evalKGProcess(
                list(preds),
                eval_tail_dict,
                all_dicts=all_tail_dicts,
                descending=eval_descending,
                num_processes=FLAGS.num_processes,
                topn=FLAGS.topn,
                queue_limit=FLAGS.max_queue,
            )
        )

        pbar.update(1)

    pbar.close()

    # hit, rank
    head_performances = [result[:2] for result in head_results]
    tail_performances = [result[:2] for result in tail_results]

    head_hit, head_mean_rank = np.array(head_performances).mean(axis=0)

    tail_hit, tail_mean_rank = np.array(tail_performances).mean(axis=0)

    logger.info("head hit:{:.4f}, head mean rank:{:.4f}, topn:{}.".format(head_hit, head_mean_rank, FLAGS.topn))

    logger.info("tail hit:{:.4f}, tail mean rank:{:.4f}, topn:{}.".format(tail_hit, tail_mean_rank, FLAGS.topn))

    head_num = len(head_results)
    tail_num = len(tail_results)

    avg_hit = float(head_hit * head_num + tail_hit * tail_num) / (head_num + tail_num)
    avg_mean_rank = float(head_mean_rank * head_num + tail_mean_rank * tail_num) / (head_num + tail_num)

    logger.info("avg hit:{:.4f}, avg mean rank:{:.4f}, topn:{}.".format(avg_hit, avg_mean_rank, FLAGS.topn))

    if is_report:
        for result in head_results:
            hit = result[0]
            rank = result[1]
            t = result[2][0]
            r = result[2][1]
            gold_h = result[3]
            logger.info("H\t{}\t{}\t{}\t{}".format(gold_h, t, r, hit))
        for result in tail_results:
            hit = result[0]
            rank = result[1]
            h = result[2][0]
            r = result[2][1]
            gold_t = result[3]
            logger.info("T\t{}\t{}\t{}\t{}".format(h, gold_t, r, hit))
    model.enable_grad()
    return avg_hit, avg_mean_rank


def train_loop(
    FLAGS, model, trainer, train_dataset, eval_datasets, entity_total, relation_total, logger, vis=None, is_report=False
):
    # 1. Setup Data
    train_iter, train_total, train_list, train_head_dict, train_tail_dict = train_dataset
    kg_dataset = KGDataset(train_list, entity_total, train_head_dict, train_tail_dict)

    # num_workers=8 is the standard "sweet spot"
    train_loader = DataLoader(kg_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    total_epochs = 100
    eval_interval = 20
    total_loss = 0.0

    # Calculate total steps for the entire training run
    total_steps = len(train_loader) * total_epochs
    interval_steps = len(train_loader) * eval_interval

    logger.info(f"Starting Training: {total_epochs} epochs total.")

    # One big pbar for the whole process
    pbar = tqdm(total=total_steps, desc="Overall Training")

    model.train()

    for epoch in range(1, total_epochs + 1):
        for batch in train_loader:
            batch = batch.to("cuda" if USE_CUDA else "cpu", non_blocking=True)
            ph, pt, pr, nh, nt, nr = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3], batch[:, 4], batch[:, 5]

            trainer.optimizer_zero_grad()

            # Forward pass
            pos_score = model(ph, pt, pr)
            neg_score = model(nh, nt, nr)
            losses = loss.marginLoss()(pos_score, neg_score, FLAGS.margin)

            # TransH specific logic & Regularization
            ent_embeddings = model.ent_embeddings(torch.cat([ph, pt, nh, nt]))
            rel_embeddings = model.rel_embeddings(torch.cat([pr, nr]))

            if FLAGS.model_type == "transh":
                norm_embeddings = model.norm_embeddings(torch.cat([pr, nr]))
                losses += loss.orthogonalLoss(rel_embeddings, norm_embeddings)

            losses = losses + loss.normLoss(ent_embeddings) + loss.normLoss(rel_embeddings)

            # Backward & Step
            losses.backward()
            nn.utils.clip_grad_norm_(model.parameters(), FLAGS.clipping_max_value)
            trainer.optimizer_step()

            batch_loss = losses.item()
            total_loss += batch_loss

            # Update pbar stats
            pbar.update(1)
            pbar.set_description(f"Epoch {epoch}/{total_epochs}")
            pbar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})

        if epoch % eval_interval == 0:
            # Calculate average loss over the 20-epoch interval
            avg_interval_loss = total_loss / interval_steps

            # Use pbar.write to avoid messing up the progress bar visual
            pbar.write(f"==> Epoch {epoch} complete. Avg Interval Loss: {avg_interval_loss:.4f}")

            # Evaluate
            trainer.my_new_performance()

            # Reset interval tracker
            total_loss = 0.0

            # Ensure model returns to training state
            model.train()
            model.enable_grad()

    pbar.close()
    trainer.save(trainer.checkpoint_path + "_final")


def run(only_forward=False):
    if FLAGS.seed != 0:  # 3
        random.seed(FLAGS.seed)  # seed(n)表示生成相同的随机数序列, 方便下次复现实验结果
        torch.manual_seed(FLAGS.seed)

    # set visualization
    vis = None
    if FLAGS.has_visualization:
        vis = Visualizer(env=FLAGS.experiment_name, port=FLAGS.visualization_port)
        vis.log(json.dumps(FLAGS.FlagValuesDict(), indent=4, sort_keys=True), win_name="Parameter")

    # set logger
    log_file = os.path.join(FLAGS.log_path, FLAGS.experiment_name + ".log")
    logger = logging.getLogger()
    log_level = logging.DEBUG if FLAGS.log_level == "debug" else logging.INFO
    logger.setLevel(level=log_level)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # FileHandler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # StreamHandler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Flag Values:\n" + json.dumps(FLAGS.FlagValuesDict(), indent=4, sort_keys=True))

    # load data
    kg_path = os.path.join(os.path.join(FLAGS.data_path, FLAGS.dataset), FLAGS.version)
    eval_files = []
    if FLAGS.kg_test_files:
        eval_files = FLAGS.kg_test_files.split(":")
    # 三元组数量
    train_dataset, eval_datasets, e_map, r_map = load_data(
        kg_path, eval_files, FLAGS.batch_size, logger=logger, negtive_samples=FLAGS.negtive_samples
    )
    entity_total = max(len(e_map), max(e_map.values()))
    relation_total = max(len(r_map), max(r_map.values()))

    train_iter, train_total, train_list, train_head_dict, train_tail_dict = train_dataset

    model = init_model(FLAGS, 0, 0, entity_total, relation_total, logger)
    epoch_length = math.ceil(train_total / FLAGS.batch_size)
    trainer = ModelTrainer(model, logger, epoch_length, FLAGS)

    # todo : load ckpt full path
    if FLAGS.load_ckpt_file is not None:  # 所以这里是用transE训练出的embedding来预训练咯？
        trainer.loadEmbedding(os.path.join(FLAGS.log_path, FLAGS.load_ckpt_file), model.state_dict(), cpu=not USE_CUDA)
        model.is_pretrained = True

    # Do an evaluation-only run.
    if only_forward:
        # head_iter, tail_iter, eval_total, eval_list, eval_head_dict, eval_tail_dict
        for i, eval_data in enumerate(eval_datasets):
            all_head_dicts = None
            all_tail_dicts = None
            if FLAGS.filter_wrong_corrupted:
                all_head_dicts = [train_head_dict] + [tmp_data[4] for j, tmp_data in enumerate(eval_datasets) if j != i]
                all_tail_dicts = [train_tail_dict] + [tmp_data[5] for j, tmp_data in enumerate(eval_datasets) if j != i]
            evaluate(
                FLAGS,
                model,
                entity_total,
                relation_total,
                eval_data[0],
                eval_data[1],
                eval_data[4],
                eval_data[5],
                all_head_dicts,
                all_tail_dicts,
                logger,
                eval_descending=False,
                is_report=FLAGS.is_report,
            )
    else:
        train_loop(
            FLAGS,
            model,
            trainer,
            train_dataset,
            eval_datasets,
            entity_total,
            relation_total,
            logger,
            vis=vis,
            is_report=False,
        )
    if vis is not None:
        vis.log("Finish!", win_name="Best Performances")


if __name__ == "__main__":
    get_flags()

    # Parse command line flags.
    FLAGS(sys.argv)
    flag_defaults(FLAGS)

    run(only_forward=FLAGS.eval_only_mode)
