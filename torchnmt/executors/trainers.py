import numpy as np
import torch
import tqdm
import sys
import time

from .base import Executor
from .validator import NMTValidator
from .utils import CheckpointSaver


class Trainer(Executor):
    def __init__(self, opts, val_opts):
        super().__init__(opts)

        self.seed = self.set_seed()
        self.saver = CheckpointSaver(root='ckpt/{}'.format(self.opts.name), seed=self.seed)

        # Dataloader
        self.dl = self.create_data_loader('train')

        # Model
        self.model = super().create_model()
        self.model.cuda()
        # self.model = nn.DataParallel(self.model)

        # Evaluator
        self.evaluator = NMTValidator(models=[self.model], opts=val_opts)

        # Hyper
        self.lr = self.opts.lr
        self.early_stop = 0
        self.optimizer = self.create_optimizer()
        print('\033[1m\033[91mModel {} created \033[0m'.format(type(self.model).__name__))
        print(self.model)

    def create_optimizer(self):
        if hasattr(self.opts, 'optimizer'):
            optimizer = self.opts.optimizer
        else:
            optimizer = 'adam'

        params = self.model.parameters()

        if optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.opts.lr, weight_decay=self.opts.weight_decay)
        elif optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.lr, weight_decay=self.opts.weight_decay)
        else:
            raise Exception("Unknown optimizer {}.".format(optimizer))

        return optimizer

    def epoch_iter(self):
        self.iteration = 0
        self.loss = 0
        self.epoch = 0
        self.current_patience = 0

        for self.epoch in range(0, self.opts.epochs + 1):
            yield

    def iteration_iter(self):
        self.pbar = tqdm.tqdm(self.dl, total=len(self.dl))
        for batch in self.pbar:
            self.pbar.set_description(
                'Epoch {}, Lr {}, Loss {:.2f}, ROUGE {:.2f}, ES {}'.format(
                    self.epoch + 1,
                    self.lr,
                    self.loss,
                    self.evaluator.best_rouge,
                    self.early_stop,
                ))
            self.batch = batch
            yield
            self.iteration += 1
            # if self.iteration == 1:
            #     break


    def update_lr(self):
        lr = self.opts.lr * 0.95 ** (self.epoch // 2)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.lr = lr

    def update_lr_plateau(self):
        if self.lr == self.opts.lr_min:
            return

        self.current_patience += 1

        # Apply decay if applicable
        if self.current_patience == self.opts.lr_decay_patience:
            lr = self.lr * self.opts.lr_decay_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
            self.current_patience = 0


class NMTTrainer(Trainer):
    def __init__(self, opts, val_opts):
        super().__init__(opts=opts, val_opts=val_opts)

    def on_epoch_start(self):
        self.losses = []
        self.model.train()

    def update(self):
        self.optimizer.zero_grad()

        time_f = time.time()
        loss = self.model(**self.batch)['loss']
        loss.backward()
        self.optimizer.step()
        self.time_forward = time.time() - time_f

        self.loss = loss.item()
        self.losses.append(self.loss)

    def on_epoch_end(self):
        loss = np.mean(self.losses)
        ppl = np.exp(loss)

        print('Avg:\tloss: {:.4g}, ppl: {:.4g}'.format(loss,
                                                       ppl))

        # self.update_lr()
        self.evaluator.epoch = self.epoch
        self.evaluator.start()

        # Fetch eval score and compute early stop
        mean_rouge = np.mean([s['ROUGE'] for s in self.evaluator.scores])
        if mean_rouge > self.evaluator.best_rouge:
            self.saver.save(model=self.model.state_dict(), tag=mean_rouge, global_step=self.epoch)
            self.evaluator.best_rouge = mean_rouge
            self.early_stop = 0
            self.current_patience = 0
        else:
            self.early_stop += 1
            self.update_lr_plateau()
        if self.early_stop == self.opts.early_stop:
            print("Early stopped reached")
            sys.exit()
