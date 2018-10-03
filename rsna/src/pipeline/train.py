from kfolds import KFold


def train_kfold(model, kfolds, folder, epochs):
	for fold_id in range(kfolds.num_folds):
		trn_images, trn_metadata, trn_labels = kfolds.get_trn_fold(fold_id)
		val_images, val_metadata, val_labels = kfolds.get_val_fold(fold_id)
		trn_gen = RSNAGenerator(folder, trn_images, trn_labels, batch_size=32, image_size=256, shuffle=True, augment=True, predict=False)
		val_gen = RSNAGenerator(folder, val_images, val_labels, batch_size=32, image_size=256, shuffle=False, predict=False)


def reset_weights(model):
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)