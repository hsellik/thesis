import os
import sys
import numpy

sys.path.append('.')

import argparse
import yaml

import numpy as np
import tensorflow as tf
import wandb

from checkpoint_tracker import Tracker
from data import data_loader, vocabulary, gpu_selector
from meta_model import VarMisuseModel
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def main():
	wandb.init(project="msc_thesis_hendrig")
	ap = argparse.ArgumentParser()
	ap.add_argument("data_path", help="Path to data root")
	ap.add_argument("vocabulary_path", help="Path to vocabulary file")
	ap.add_argument("config", help="Path to config file")
	ap.add_argument("-m", "--models", help="Directory to store trained models (optional)")
	ap.add_argument("-l", "--log", help="Path to store training log (optional)")
	ap.add_argument("-e", "--eval_only", help="Whether to run just the final model evaluation")
	args = ap.parse_args()
	config = yaml.safe_load(open(args.config))
	wandb.config.update(args)
	wandb.config.update(config)
	lowest_memory_gpu = gpu_selector.GPUSelector().pick_gpu_lowest_memory()
	if lowest_memory_gpu is not None:
		os.environ["CUDA_VISIBLE_DEVICES"] = str(lowest_memory_gpu)
	print("Training with configuration:", config)
	data = data_loader.DataLoader(args.data_path, config["data"], vocabulary.Vocabulary(args.vocabulary_path))
	if args.eval_only:
		if args.models is None or args.log is None:
			raise ValueError("Must provide a path to pre-trained models when running final evaluation")
		test(data, config, args.models, args.log)
	else:
		train(data, config, args.models, args.log)

def test(data, config, model_path, log_path):
	model = VarMisuseModel(config['model'], data.vocabulary.vocab_dim)
	model.run_dummy_input()
	tracker = Tracker(model, model_path, log_path)
	tracker.restore(best_model=True)
	evaluate(data, config, model, is_heldout=False)

def train(data, config, model_path=None, log_path=None):
	model = VarMisuseModel(config['model'], data.vocabulary.vocab_dim)
	model.run_dummy_input()
	print("Model initialized, training {:,} parameters".format(np.sum([np.prod(v.shape) for v in model.trainable_variables])))
	optimizer = tf.optimizers.Adam(config["training"]["learning_rate"])

	# Restore model from checkpoints if present; also sets up logger
	if model_path is None:
		tracker = Tracker(model)
	else:
		tracker = Tracker(model, model_path, log_path)
	tracker.restore()
	if tracker.ckpt.step.numpy() > 0:
		print("Restored from step:", tracker.ckpt.step.numpy() + 1)
	else:
		print("Step:", tracker.ckpt.step.numpy() + 1)

	mbs = 0
	losses, accs, counts = get_metrics()
	while tracker.ckpt.step < config["training"]["max_steps"]:
		# These are just for console logging, not global counts
		for batch in data.batcher(mode='train'):
			mbs += 1
			tokens, edges, error_loc, repair_targets, repair_candidates = batch
			token_mask = tf.clip_by_value(tf.reduce_sum(tokens, -1), 0, 1)

			with tf.GradientTape() as tape:
				pointer_preds = model(tokens, token_mask, edges, training=True)
				ls, acs, binary_data = model.get_loss(pointer_preds, token_mask, error_loc, repair_targets, repair_candidates)
				y_true, y_pred = binary_data

				loc_loss, rep_loss = ls
				loss = loc_loss + rep_loss

			grads = tape.gradient(loss, model.trainable_variables)
			grads, _ = tf.clip_by_global_norm(grads, 0.25)
			optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))

			# Update statistics
			num_buggy = tf.reduce_sum(tf.clip_by_value(error_loc, 0, 1))
			samples = tf.shape(token_mask)[0]
			prev_samples = tracker.get_samples()
			curr_samples = tracker.update_samples(samples)
			update_metrics(losses, accs, counts, token_mask, ls, acs, num_buggy)

			# Every few minibatches, print the recent training performance
			if mbs % config["training"]["print_freq"] == 0:
				avg_losses = ["{0:.3f}".format(l.result().numpy()) for l in losses]
				avg_accs = ["{0:.2%}".format(a.result().numpy()) for a in accs]
				precision = precision_score(y_true, y_pred, average='binary')
				recall = recall_score(y_true, y_pred, average='binary')
				print(f"MB: {mbs}, seqs: {curr_samples:,}, tokens: {counts[1].result().numpy():,}, loss: {losses[0].result().numpy()}, no_bug_pred_acc: {avg_accs[0]}, bug_loc_acc: {avg_accs[1]}, precision: {precision}, recall: {recall}")
				wandb.log({'loss': losses[0].result().numpy(), 'no_bug_pred_acc': accs[0].result().numpy(), 'bug_loc_acc': accs[1].result().numpy(), 'precision': precision, 'recall': recall})
				[l.reset_states() for l in losses]
				[a.reset_states() for a in accs]
			
			# Every valid_interval samples, run an evaluation pass and store the most recent model with its heldout accuracy
			if prev_samples // config["data"]["valid_interval"] < curr_samples // config["data"]["valid_interval"]:
				avg_accs = evaluate(data, config, model)
				tracker.save_checkpoint(model, avg_accs)
				if tracker.ckpt.step >= config["training"]["max_steps"]:
					break
				else:
					print("Step:", tracker.ckpt.step.numpy() + 1)

def evaluate(data, config, model, is_heldout=True):  # Similar to train, just without gradient updates
	if is_heldout:
		print("Running evaluation pass on heldout data")
	else:
		print("Testing pre-trained model on full eval data")
	
	losses, accs, counts = get_metrics()
	mbs = 0
	y_true_all = []
	y_pred_all = []
	for batch in data.batcher(mode='dev' if is_heldout else 'eval'):
		mbs += 1
		tokens, edges, error_loc, repair_targets, repair_candidates = batch		
		token_mask = tf.clip_by_value(tf.reduce_sum(tokens, -1), 0, 1)
		
		pointer_preds = model(tokens, token_mask, edges, training=False)
		ls, acs, binary_data = model.get_loss(pointer_preds, token_mask, error_loc, repair_targets, repair_candidates)
		y_true, y_pred = binary_data
		y_true_all = numpy.append(y_true_all, y_true.numpy())
		y_pred_all = numpy.append(y_pred_all, y_pred.numpy())
		precision = precision_score(y_true, y_pred, average='binary')
		recall = recall_score(y_true, y_pred, average='binary')
		num_buggy = tf.reduce_sum(tf.clip_by_value(error_loc, 0, 1))
		update_metrics(losses, accs, counts, token_mask, ls, acs, num_buggy)
		if is_heldout and counts[0].result() > config['data']['max_valid_samples']:
			break
		if not is_heldout and mbs % config["training"]["print_freq"] == 0:
			avg_losses = ["{0:.3f}".format(l.result().numpy()) for l in losses]
			avg_accs = ["{0:.2%}".format(a.result().numpy()) for a in accs]
			print(f"MB: {mbs}, seqs: {counts[0].result().numpy():,}, tokens: {counts[1].result().numpy():,}, loss: {losses[0].result().numpy()}, no_bug_pred_acc: {avg_accs[0]}, bug_loc_acc: {avg_accs[1]}, precision: {precision}, recall: {recall}")

	avg_accs = [a.result().numpy() for a in accs]
	avg_accs_str = ", ".join(["{0:.2%}".format(a) for a in avg_accs])
	avg_loss_str = ", ".join(["{0:.3f}".format(l.result().numpy()) for l in losses])
	# print(f"Old evaluation result: seqs: {counts[0].result().numpy():,}, tokens: {counts[1].result().numpy():,}, loss: {avg_loss_str}, accs: {avg_accs_str}")
	precision = precision_score(y_true_all, y_pred_all, average='binary')
	recall = recall_score(y_true_all, y_pred_all, average='binary')
	wandb.log({'loss': losses[0].result().numpy(), 'no_bug_pred_acc': accs[0].result().numpy(), 'bug_loc_acc': accs[1].result().numpy(), 'precision': precision, 'recall': recall})
	print(f"New evaluation results: seqs: {counts[0].result().numpy():,}, tokens: {counts[1].result().numpy():,}, loss: {losses[0].result().numpy()}, no_bug_pred_acc: {avg_accs[0]}, bug_loc_acc: {avg_accs[1]}, precision: {precision}, recall: {recall}")
	return avg_accs

def get_metrics():
	losses = [tf.keras.metrics.Mean() for _ in range(2)]
	accs = [tf.keras.metrics.Mean() for _ in range(4)]
	counts = [tf.keras.metrics.Sum(dtype='int32') for _ in range(2)]
	return losses, accs, counts

def update_metrics(losses, accs, counts, token_mask, ls, acs, num_buggy_samples):
	loc_loss, rep_loss = ls
	no_bug_pred_acc, bug_loc_acc, target_loc_acc, joint_acc = acs
	num_samples = tf.shape(token_mask)[0]
	counts[0].update_state(num_samples)
	counts[1].update_state(tf.reduce_sum(token_mask))
	losses[0].update_state(loc_loss)
	losses[1].update_state(rep_loss)
	accs[0].update_state(no_bug_pred_acc, sample_weight=num_samples - num_buggy_samples)
	accs[1].update_state(bug_loc_acc, sample_weight=num_buggy_samples)
	accs[2].update_state(target_loc_acc, sample_weight=num_buggy_samples)
	accs[3].update_state(joint_acc, sample_weight=num_buggy_samples)

if __name__ == '__main__':
	main()