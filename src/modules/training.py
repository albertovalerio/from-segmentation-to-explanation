"""
Functions for training the models.
"""
import os, random, time, calendar, datetime
from sys import platform
import torch
import pandas as pd
import numpy as np
import nibabel as nib
from monai.data import DataLoader, decollate_batch, CacheDataset
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.handlers.utils import from_engine
from src.helpers.utils import get_date_time, save_results


__all__ = ['train_test_splitting', 'training_model', 'predict_model']


def train_test_splitting(
		folder,
		train_ratio=.8,
		reports_path=None,
		write_to_file=False,
		load_from_file=False,
		verbose=True
	):
	"""
	Splitting train/eval/test.
	Args:
		folder (str): the path of the folder containing data.
		train_ratio (float): ratio of the training set, value between 0 and 1.
		reports_path (str): folder where to save report file.
			Required if `write_to_file` is True and/or `load_from_file` is True.
		write_to_file (bool): whether to write selected data to csv file.
		load_from_file (bool): whether to load splitting data from a previous saved csv file.
		verbose (bool): whether or not print information.
	Returns:
		train_data (list): the training data ready to feed monai.data.Dataset
		eval_data (list): the evaluation data ready to feed monai.data.Dataset
		test_data (list): the testing data ready to feed monai.data.Dataset.
		(see https://docs.monai.io/en/latest/data.html#monai.data.Dataset).
	"""
	sessions = ['-'.join(s.split('-')[:3]) for s in os.listdir(folder) if os.path.isdir(os.path.join(folder, s))]
	subjects = list(set(sessions))
	random.shuffle(subjects)
	split_train = int(len(subjects) * train_ratio)
	train_subjects, test_subjects = subjects[:split_train], subjects[split_train:]
	split_eval = int(len(train_subjects) * .8)
	eval_subjects = train_subjects[split_eval:]
	train_subjects = train_subjects[:split_eval]
	if load_from_file:
		if not reports_path:
			print('\n' + ''.join(['> ' for i in range(30)]))
			print('\nERROR: Paremeter \033[95m `reports_path`\033[0m must be specified.\n')
			print(''.join(['> ' for i in range(30)]) + '\n')
			return [],[],[]
		else:
			df_split = pd.read_csv(os.path.join(reports_path, [i for i in os.listdir(reports_path) if 'splitting' in i][0]))
			train_subjects = df_split['train_subjects'].dropna().to_numpy()
			eval_subjects = df_split['eval_subjects'].dropna().to_numpy()
			test_subjects = df_split['test_subjects'].dropna().to_numpy()
	if write_to_file:
		if not reports_path:
			print('\n' + ''.join(['> ' for i in range(30)]))
			print('\nERROR: Paremeter \033[95m `reports_path`\033[0m must be specified.\n')
			print(''.join(['> ' for i in range(30)]) + '\n')
			return [],[],[]
		else:
			for i in range(max(len(train_subjects), len(eval_subjects), len(test_subjects))):
				save_results(
					os.path.join(reports_path, 'splitting_'+str(calendar.timegm(time.gmtime()))+'.csv'),
					{
						'train_subjects': train_subjects[i] if i < len(train_subjects) else '',
						'eval_subjects': eval_subjects[i] if i < len(eval_subjects) else '',
						'test_subjects': test_subjects[i] if i < len(test_subjects) else ''
					}
				)
	train_sessions = [os.path.join(folder, s) for s in os.listdir(folder) if '-'.join(s.split('-')[:3]) in train_subjects]
	eval_sessions = [os.path.join(folder, s) for s in os.listdir(folder) if '-'.join(s.split('-')[:3]) in eval_subjects]
	test_sessions = [os.path.join(folder, s) for s in os.listdir(folder) if '-'.join(s.split('-')[:3]) in test_subjects]
	train_labels = [os.path.join(s, s.split('/')[-1] + '-seg.nii.gz') for s in train_sessions]
	eval_labels = [os.path.join(s, s.split('/')[-1] + '-seg.nii.gz') for s in eval_sessions]
	test_labels = [os.path.join(s, s.split('/')[-1] + '-seg.nii.gz') for s in test_sessions]
	modes = ['t1c', 't1n', 't2f', 't2w']
	train_data, eval_data, test_data = {}, {}, {}
	train_data = [dict({
		'image': [os.path.join(s, s.split('/')[-1] + '-' + m + '.nii.gz') for m in modes],
		'label': train_labels[i],
		'subject': s.split('/')[-1]
	}) for i, s in enumerate(train_sessions)]
	eval_data = [dict({
		'image': [os.path.join(s, s.split('/')[-1] + '-' + m + '.nii.gz') for m in modes],
		'label': eval_labels[i],
		'subject': s.split('/')[-1]
	}) for i, s in enumerate(eval_sessions)]
	test_data = [dict({
		'image': [os.path.join(s, s.split('/')[-1] + '-' + m + '.nii.gz') for m in modes],
		'label': test_labels[i],
		'subject': s.split('/')[-1]
	}) for i, s in enumerate(test_sessions)]
	if verbose:
		print(''.join(['> ' for i in range(40)]))
		print(f'\n{"":<20}{"TRAINING":<20}{"EVALUATION":<20}{"TESTING":<20}\n')
		print(''.join(['> ' for i in range(40)]))
		tsb1 = str(len(train_subjects)) + ' (' + str(round((len(train_subjects) * 100 / len(subjects)), 0)) + ' %)'
		tsb2 = str(len(eval_subjects)) + ' (' + str(round((len(eval_subjects) * 100 / len(subjects)), 0)) + ' %)'
		tsb3 = str(len(test_subjects)) + ' (' + str(round((len(test_subjects) * 100 / len(subjects)), 0)) + ' %)'
		tss1 = str(len(train_sessions)) + ' (' + str(round((len(train_sessions) * 100 / len(sessions)), 2)) + ' %)'
		tss2 = str(len(eval_sessions)) + ' (' + str(round((len(eval_sessions) * 100 / len(sessions)), 2)) + ' %)'
		tss3 = str(len(test_sessions)) + ' (' + str(round((len(test_sessions) * 100 / len(sessions)), 2)) + ' %)'
		print(f'\n{"subjects":<20}{tsb1:<20}{tsb2:<20}{tsb3:<20}\n')
		print(f'{"sessions":<20}{tss1:<20}{tss2:<20}{tss3:<20}\n')
	return train_data, eval_data, test_data


def training_model(
		model,
		data,
		transforms,
		epochs,
		device,
		paths,
		val_interval=1,
		early_stopping=10,
		num_workers=4,
		ministep=12,
		write_to_file=True,
		verbose=False
	):
	"""
	Standard Pytorch-style training program.
	Args:
		model (torch.nn.Module): the model to be trained.
		data (list): the training and evalutaion data.
		transform (list): transformation sequence for training and evaluation data.
		epochs (int): max number of epochs.
		device (str): device's name.
		paths (list): folders where to save results and model's dump.
		val_interval (int): validation interval.
		early_stopping (int): nr. of epochs for those there's no more improvements.
		num_workers (int): setting multi-process data loading.
		ministep (int): number of interval of data to load on RAM.
		write_to_file (bool): whether to write results to csv file.
		verbose (bool): whether to print minimal or extended information.
	Returns:
		metrics (list): the list of all the computed metrics over the training in this order:
			- dice loss during training;
			- dice loss during evaluation;
			- execution times;
			- average dice score;
			- dice score for the class ET;
			- dice score for the class TC;
			- dice score for the class WT.
	"""
	# unfolds grouped data/init model and utils
	device = torch.device(device)
	model = model.to(device)
	train_data, eval_data = data
	train_transform, eval_transform, post_trans = transforms
	saved_path, reports_path, logs_path = paths
	ministep = ministep if (len(train_data) > 10 and len(eval_data) > 10 and ministep > 1) else 2

	# define Dice loss, Adam optimizer, mean Dice metric, Cosine Annealing scheduler
	loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
	optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
	lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
	dice_metric = DiceMetric(include_background=True, reduction='mean')
	dice_metric_batch = DiceMetric(include_background=True, reduction='mean_batch')
	scaler = torch.cuda.amp.GradScaler() # use Automatic Pixed Precision to accelerate training
	torch.backends.cudnn.benchmark = True # enable cuDNN benchmark

	# define metric/loss collectors
	best_metric, best_metric_epoch = -1, -1
	best_metrics_epochs_and_time = [[], [], []]
	epoch_loss_values, epoch_time_values = [[], []], []
	metric_values, metric_values_et, metric_values_tc, metric_values_wt = [], [], [], []

	# log the current execution
	log = open(os.path.join(logs_path, 'training.log'), 'a', encoding='utf-8')
	log.write('['+get_date_time()+'] Training phase started.EXECUTING: ' + model.name + '\n')
	log.flush()
	ts = calendar.timegm(time.gmtime())
	total_start = time.time()
	for epoch in range(epochs):
		epoch_start = time.time()
		print(''.join(['> ' for i in range(40)]))
		print(f"epoch {epoch + 1}/{epochs}")
		log.write('['+get_date_time() + '] EXECUTING.' + model.name + ' EPOCH ' + str(epoch + 1) + ' OF ' + str(epochs) + ' \n')
		log.flush()
		model.train()
		epoch_loss_train, epoch_loss_eval = 0, 0
		step_train, step_eval = 0, 0
		ministeps_train = np.linspace(0, len(train_data), ministep).astype(int)
		ministeps_eval = np.linspace(0, len(eval_data), ministep).astype(int)

		# start training
		for i in range(len(ministeps_train) - 1):
			train_ds = CacheDataset(train_data[ministeps_train[i]:ministeps_train[i+1]], transform=train_transform, cache_rate=1.0, num_workers=None, progress=False)
			train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=num_workers)
			for batch_data in train_loader:
				step_start = time.time()
				step_train += 1
				inputs, labels = (batch_data['image'].to(device), batch_data['label'].to(device))
				optimizer.zero_grad()
				with torch.cuda.amp.autocast():
					outputs = model(inputs)
					loss = loss_function(outputs, labels)
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
				epoch_loss_train += loss.item()
				if verbose:
					print(
						f"{step_train}/{len(train_data) // train_loader.batch_size}"
						f", train_loss: {loss.item():.4f}"
						f", step time: {str(datetime.timedelta(seconds=int(time.time() - step_start)))}"
					)
		lr_scheduler.step()
		epoch_loss_train /= step_train
		epoch_loss_values[0].append(epoch_loss_train)
		print(f"epoch {epoch + 1} average training loss: {epoch_loss_train:.4f}")

		# start validation
		if (epoch + 1) % val_interval == 0:
			model.eval()
			with torch.no_grad():
				for i in range(len(ministeps_eval) - 1):
					eval_ds = CacheDataset(eval_data[ministeps_eval[i]:ministeps_eval[i+1]], transform=eval_transform, cache_rate=1.0, num_workers=None, progress=False)
					eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=True, num_workers=num_workers)
					for val_data in eval_loader:
						step_eval += 1
						val_inputs, val_labels = (val_data['image'].to(device), val_data['label'].to(device))
						val_outputs = inference(val_inputs, device, model)
						val_loss = loss_function(val_outputs, val_labels)
						epoch_loss_eval += val_loss.item()
						val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
						dice_metric(y_pred=val_outputs, y=val_labels)
						dice_metric_batch(y_pred=val_outputs, y=val_labels)
				epoch_loss_eval /= step_eval
				epoch_loss_values[1].append(epoch_loss_eval)

				# calculate metrics
				metric = dice_metric.aggregate().item()
				metric_values.append(metric)
				metric_batch = dice_metric_batch.aggregate()
				metric_et = metric_batch[0].item()
				metric_values_et.append(metric_et)
				metric_tc = metric_batch[1].item()
				metric_values_tc.append(metric_tc)
				metric_wt = metric_batch[2].item()
				metric_values_wt.append(metric_wt)
				dice_metric.reset()
				dice_metric_batch.reset()

				# save best performing model
				if metric > best_metric:
					best_metric = metric
					best_metric_epoch = epoch + 1
					best_metrics_epochs_and_time[0].append(best_metric)
					best_metrics_epochs_and_time[1].append(best_metric_epoch)
					best_metrics_epochs_and_time[2].append(time.time() - total_start)
					torch.save(model.state_dict(), os.path.join(saved_path, model.name + '_best.pth'))
					print("saved new best model")
				print(
					f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
					f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
					f"\nbest mean dice: {best_metric:.4f}"
					f" at epoch: {best_metric_epoch}"
				)
		print(f"time consuming of epoch {epoch + 1} is: {str(datetime.timedelta(seconds=int(time.time() - epoch_start)))}")
		epoch_time_values.append(time.time() - epoch_start)

		# save results to file
		if write_to_file:
			save_results(
				file = os.path.join(reports_path, model.name + '_training.csv'),
				metrics = {
					'id': model.name.upper() + '_' + str(ts),
					'epoch': epoch + 1,
					'model': model.name,
					'train_dice_loss': epoch_loss_train,
					'eval_dice_loss': epoch_loss_eval,
					'exec_time': time.time() - epoch_start,
					'dice_score': metric,
					'dice_score_et': metric_et,
					'dice_score_tc': metric_tc,
					'dice_score_wt': metric_wt,
					'datetime': get_date_time()
				}
			)

		# early stopping
		if epoch + 1 - best_metric_epoch == early_stopping:
			print(f"\nEarly stopping triggered at epoch: {str(epoch + 1)}\n")
			break

	print(f"\n\nTrain completed! Best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {str(datetime.timedelta(seconds=int(time.time() - total_start)))}.")
	log.write('['+get_date_time()+'] Training phase ended.EXECUTING: ' + model.name + '\n')
	log.flush()
	log.close()
	return [
		epoch_loss_values[0],
		epoch_loss_values[1],
		epoch_time_values,
		metric_values,
		metric_values_et,
		metric_values_tc,
		metric_values_wt
	]


def predict_model(
	model,
	data,
	transforms,
	device,
	paths,
	ministep=4,
	num_workers=4,
	write_to_file=True,
	save_sample=3,
	return_prediction=False,
	verbose=False
):
	"""
	Standard Pytorch-style prediction program.
	Args:
		model (torch.nn.Module): the model to be loaded.
		data (list): the testing data.
		transform (list): pre and post transformations for testing data.
		device (str): device's name.
		paths (list): folders where to save results and load model's dump.
		num_workers (int): setting multi-process data loading.
		ministep (int): number of interval of data to load on RAM.
		write_to_file (bool): whether to write results to csv file.
		save_sample (int): number of predicted image samples to save into `predictions` folder.
		return_prediction (bool): whether or not return the predicted mask.
		verbose (bool): whether or not print information.
	Returns:
		metrics (list): dice score and Hausdorff distance for each class.
	"""
	# unfolds grouped data/init model and utils
	device = torch.device(device)
	model = model.to(device)
	predictions, counter = [], 0
	subjects = sorted([t['subject'] for t in data])
	samples_random = random.sample(subjects, save_sample)
	ministep = ministep if (len(data) > 5 and ministep > 1) else 2
	ministeps_test = np.linspace(0, len(data), ministep).astype(int)
	test_transform, post_test_transforms = transforms
	saved_path, reports_path, preds_path, logs_path = paths

	# define metrics
	dice_metric_batch = DiceMetric(include_background=True, reduction='mean_batch')
	hausdorff_metric_batch = HausdorffDistanceMetric(include_background=True, reduction='mean_batch', percentile=95)

	# log the current execution
	log = open(os.path.join(logs_path, 'prediction.log'), 'a', encoding='utf-8')
	log.write('['+get_date_time()+'] Predictions started.EXECUTING: ' + model.name + '\n')
	log.flush()

	try:
		# load pretrained model
		model.load_state_dict(
			torch.load(os.path.join(saved_path, model.name + '_best.pth'), map_location=torch.device(device), weights_only=True)
		)
		model.eval()
		# making inference
		with torch.no_grad():
			for i in range(len(ministeps_test) - 1):
				test_ds = CacheDataset(data[ministeps_test[i]:ministeps_test[i+1]], transform=test_transform, cache_rate=1.0, num_workers=None, progress=False)
				test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers)
				for val_data in test_loader:
					val_inputs = val_data['image'].to(device)
					val_data['pred'] = inference(val_inputs, device, model)
					val_data = [post_test_transforms(i) for i in decollate_batch(val_data)]
					# save samples images
					if val_data[0]['subject'] in samples_random:
						sample = {
							'image': val_data[0]['image'],
							'label': val_data[0]['label'],
							'pred': val_data[0]['pred']
						}
						for k in sample.keys():
							nii_image = nib.Nifti1Image(sample[k].detach().cpu().numpy(), affine=np.eye(4))
							filename = model.name + '_' + val_data[0]['subject'] + '_' + k + '.nii.gz'
							nib.save(nii_image, os.path.join(preds_path, filename))
					if return_prediction:
						predictions.append({'image': val_data[0]['image'], 'label': val_data[0]['label'], 'pred': val_data[0]['pred'], 'subject': val_data[0]['subject']})
					val_outputs, val_labels = from_engine(['pred', 'label'])(val_data)
					# compute metrics
					dice_metric_batch(y_pred=val_outputs, y=val_labels)
					hausdorff_metric_batch(y_pred=val_outputs, y=val_labels)
					counter += 1
					if verbose and ((counter - 1) == 0 or (counter % int(len(data) / 5)) == 0):
						print(f"inference {counter}/{len(data)}")
						log.write('['+get_date_time()+'] EXECUTING.'+model.name+' INFERENCE '+str(counter)+' OF '+str(len(data))+' \n')
						log.flush()
			dice_batch_org = dice_metric_batch.aggregate()
			hausdorff_batch_org = hausdorff_metric_batch.aggregate()
			dice_metric_batch.reset()
			hausdorff_metric_batch.reset()
			metrics = [[i.item() for i in dice_batch_org], [j.item() for j in hausdorff_batch_org]]

			# save results to file
			if write_to_file:
				save_results(
					file = os.path.join(reports_path, 'results.csv'),
					metrics = {
						'model': model.name,
						'n_images': len(data),
						'size_images': test_ds[0]['image'][0].shape[0],
						'dice_score_et': metrics[0][0],
						'dice_score_tc': metrics[0][1],
						'dice_score_wt': metrics[0][2],
						'dice_score_mean': np.mean([metrics[0][0], metrics[0][1], metrics[0][2]]),
						'hausdorff_score_et': metrics[1][0],
						'hausdorff_score_tc': metrics[1][1],
						'hausdorff_score_wt': metrics[1][2],
						'hausdorff_score_mean': np.mean([metrics[1][0], metrics[1][1], metrics[1][2]]),
						'datetime': get_date_time()
					}
				)

			log.write('['+get_date_time()+'] Predictions ended.EXECUTING: ' + model.name + '\n')
			log.flush()
			log.close()
			return metrics, predictions
	except OSError as e:
		print('\n' + ''.join(['> ' for i in range(30)]))
		print('\nERROR: model dump for\033[95m '+model.name+'\033[0m not found.\n')
		print(''.join(['> ' for i in range(30)]) + '\n')


def inference(input, device, model):
	"""
	Define inference method.
	"""
	device = torch.device(device) if type(device) is str else device
	def _compute(input):
		return sliding_window_inference(
			inputs = input,
			roi_size = (128, 128, 128) if model.name == 'SwinUNETR' else (240, 240, 160),
			sw_batch_size = 1,
			predictor = model,
			overlap = .5,
		)
	if device.type == 'cuda':
		with torch.cuda.amp.autocast():
			return _compute(input)
	else:
		return _compute(input)
