# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train_imagenet."""
import argparse
import ast
import os
import time

from collections import OrderedDict

import mindspore
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore.ops.operations as P

from mindspore import context
from mindspore.context import ParallelMode
from mindspore import Tensor
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore.train.callback import Callback

from src.proxylessnas import proxylessnas_mobile
from src.dataset import create_dataset
from src.lr_generator import *

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_rank_id

set_seed(1)

def modelarts_pre_process():
    pass

best_acc = 0.0
best_epoch = 0
   
class EvalCallBack(Callback):
    
    def __init__(self, net , model, eval_dataset, eval_per_epochs, train_batches_per_epoch ):
        self.net    = net
        self.model  = model

        self.eval_dataset = eval_dataset
        self.eval_per_epochs = eval_per_epochs
        self.train_batches_per_epoch = train_batches_per_epoch
        self.device_id    = mindspore.context.get_context("device_id")

        if config.enable_modelarts :
           self.save_ckpt_path = config.output_path + '/' + os.getenv('DEVICE_ID') 
           self.device_0_ckpt_path = config.output_path + '/0'
        else:
           self.save_ckpt_path = config.save_ckpt_path
           self.device_0_ckpt_path = config.save_ckpt_path + '/0'
        self.log_filename = os.path.join(self.save_ckpt_path , 'log_' + str(self.device_id) + '.txt' )
           
    def epoch_end(self, run_context):
        global best_acc
        global best_epoch
        
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num

        fp = open(self.log_filename,'at+')
        
        if cur_epoch % self.eval_per_epochs == 0:
           acc = self.model.eval(self.eval_dataset, dataset_sink_mode=True)
           #print(cb_param)
           print(acc , " previous best_acc is ", int(best_acc*10000)/10000.0 , " on epoch " , best_epoch , " of device " , os.getenv('DEVICE_ID'))
           print("epoch : " , cur_epoch, acc , " previous best_acc is ", int(best_acc*10000)/10000.0 , " on epoch " , best_epoch , " of device " , os.getenv('DEVICE_ID') , file= fp )

           if acc['acc'] > best_acc :
              best_acc = acc['acc'] 
              best_epoch = cur_epoch
              #if best_acc*10000 > 1000 :
              #   checkpoint_filename = 'acc_0.' + str(int(acc['acc']*10000)) + "_epoch_" + str(cur_epoch) + "_device_" + str(self.device_id) +'.ckpt'
              #   checkpoint_filename = self.save_ckpt_path + '/' + checkpoint_filename
              #   mindspore.save_checkpoint( self.net , checkpoint_filename )            
              checkpoint_filename = 'best_acc_device_' + str(self.device_id) +'.ckpt'
              checkpoint_filename = self.save_ckpt_path + '/' + checkpoint_filename
              mindspore.save_checkpoint( self.net , checkpoint_filename )            
              print("*********************************************")
              print("*********************************************" , file= fp )
           if cur_epoch >= config.epoch_size - config.keep_checkpoint_max :
              checkpoint_filename = 'acc_0.' + str(int(acc['acc']*10000)) + "_epoch_" + str(cur_epoch) + "_device_" + str(self.device_id) +'.ckpt'
              checkpoint_filename = self.save_ckpt_path + '/' + checkpoint_filename
              if os.path.exists(checkpoint_filename) == False:
                 mindspore.save_checkpoint( self.net , checkpoint_filename )            
           if acc['acc'] >= 0.7456  and config.num_classes >= 1000 :
              print("+++++++++++++++++++++++++++++++++++++++++++++")        
              print("+++++++++++++++++++++++++++++++++++++++++++++" , file= fp )       
        if mindspore.context.get_context("device_id") != 0 :
           checkpoint_filename = "proxylessnas_mobile_device_0-" + str(cur_epoch) + "_" + str(self.train_batches_per_epoch) +'.ckpt'
           checkpoint_filename = self.device_0_ckpt_path + '/' + checkpoint_filename
           if os.path.exists(checkpoint_filename):
              ckpt = load_checkpoint( checkpoint_filename )
              load_param_into_net(self.net, ckpt , strict_load=False)
              self.net.set_train() 
           else:
              print(checkpoint_filename , " is not exist")
              print(checkpoint_filename , " is not exist" , file= fp )
        #prepare the drop_path_prob for the next epoch      
        #cur_epoch is start from 1
        #self.net.drop_path_prob = config.drop_path_prob * cur_epoch / config.epoch_size

        fp.close()



@moxing_wrapper(pre_process=modelarts_pre_process)
def train():
    global best_acc
    global best_epoch

    start_time = time.time()

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
    # init distributed
    if config.is_distributed:
        if os.getenv('DEVICE_ID', "not_set").isdigit():
            context.set_context(device_id=get_device_id())
        init()
        rank = get_rank_id()
        group_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=group_size, gradients_mean=True)
    else:
        rank = 0
        group_size = 1
        context.set_context(device_id=config.device_id)

    # define network
    net = proxylessnas_mobile(num_classes = config.num_classes)

    # define loss
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True,
                                            reduction='mean')

    # define dataset
    if config.enable_modelarts:
       mox = __import__("moxing")
       mox.file.copy_parallel(src_url=config.data_url, dst_url='/cache/data_path_' + os.getenv('DEVICE_ID'))
       train_data_path = '/cache/data_path_' + os.getenv('DEVICE_ID') + '/train'
       val_data_path = '/cache/data_path_' + os.getenv('DEVICE_ID') + '/val'
       train_dataset = create_dataset(train_data_path, do_train=True, device_num=group_size, rank=rank, batch_size=config.train_batch_size, drop_remainder = False, shuffle = True )
       val_dataset = create_dataset(val_data_path, do_train=False, device_num=group_size, rank=rank, batch_size=config.val_batch_size, drop_remainder = False, shuffle = True )

    else:
       train_dataset = create_dataset(config.data_path+'/train', do_train=True, device_num=group_size, rank=rank, batch_size=config.train_batch_size, drop_remainder = False, shuffle = True )
       val_dataset = create_dataset(config.data_path+'/val', do_train=False, device_num=group_size, rank=rank, batch_size=config.val_batch_size, drop_remainder = False, shuffle = True )
    
    train_batches_per_epoch = train_dataset.get_dataset_size()
    epoch_size = config.epoch_size
    keep_checkpoint_max = config.keep_checkpoint_max

    # get learning rate
    lr = warmup_cosine_annealing_lr(lr=config.lr_init, max_epoch=epoch_size , steps_per_epoch=train_batches_per_epoch, warmup_epochs=5, T_max=epoch_size, eta_min=0.0)
    #lr = my_warmup_cosine_annealing_lr(lr=config.lr_init, max_epoch=epoch_size , steps_per_epoch=train_batches_per_epoch, warmup_epochs=5, T_max=epoch_size-60, eta_min=0.0)

    lr = Tensor(lr)

    # define optimization
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
        #if 'beta' not in param.name and 'gamma' not in param.name :
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]

    #decayed_params_1 = [] 
    #decayed_params_2 = [] 
    #decayed_params_3 = [] 
    #decayed_params_4 = [] 
    #for param in net.trainable_params():
    #    if '.bn.gamma' in param.name :
    #        decayed_params_1.append(param)
    #    elif '.bn.beta' in param.name :
    #        decayed_params_2.append(param)
    #    elif '.bias' in param.name :
    #        decayed_params_3.append(param)
    #    else:            
    #        decayed_params_4.append(param)
    #group_params = [{'params': decayed_params_1, 'weight_decay': config.weight_decay*0.0001},
    #                {'params': decayed_params_2, 'weight_decay': config.weight_decay*0.0001},
    #                {'params': decayed_params_3, 'weight_decay': config.weight_decay*0.01},
    #                {'params': decayed_params_4, 'weight_decay': config.weight_decay},
    #                {'order_params': net.trainable_params()}]
    #                
    optimizer = Momentum(params=group_params, learning_rate=lr, momentum=config.momentum,
                         weight_decay=config.weight_decay)
    #                     
    #optimizer = Momentum(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
    #                     weight_decay=config.weight_decay)
    
    model = Model(net, loss_fn=loss, optimizer=optimizer, amp_level=config.amp_level, metrics={'acc'})

    # define callbacks
    eval_cb = EvalCallBack(net , model, val_dataset, 1 , train_batches_per_epoch )

    if config.enable_modelarts :
        save_ckpt_path = config.output_path + '/' + os.getenv('DEVICE_ID') 
    else:
        save_ckpt_path = config.save_ckpt_path
    config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * train_batches_per_epoch,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint("proxylessnas_mobile_device_"+os.getenv('DEVICE_ID'), directory=save_ckpt_path, config=config_ck)

    cb = [ ckpt_cb, TimeMonitor(), LossMonitor(),eval_cb ]

    # begin train
    print("============== Starting Training ==============")
    model.train(config.epoch_size, train_dataset, callbacks=cb, dataset_sink_mode=True)

    print("data_url   = " , config.data_url ) 
    print("epoch_size = " , epoch_size , " train_batch_size = " , config.train_batch_size , " lr_init = " , config.lr_init  , " weight_decay = " , config.weight_decay ) 
    print("best_acc is ", int(best_acc*10000)/10000.0 , " on epoch " , best_epoch , " of device " , os.getenv('DEVICE_ID'))
    print("time: ", (time.time() - start_time) / 3600)
    print("============== Train Success ==================")

    device_id    = mindspore.context.get_context("device_id")
    log_filename = os.path.join(save_ckpt_path , 'log_' + str(device_id) + '.txt' )

    fp = open(log_filename,'at+')

    print("data_url   = " , config.data_url  , file= fp ) 
    print("epoch_size = " , epoch_size , " train_batch_size = " , config.train_batch_size , " lr_init = " , config.lr_init  , " weight_decay = " , config.weight_decay, file= fp ) 
    print("best_acc is ", int(best_acc*10000)/10000.0 , " on epoch " , best_epoch , " of device " , os.getenv('DEVICE_ID'), file= fp )     
    print("time: ", (time.time() - start_time) / 3600, file= fp ) 

    fp.close()
    
if __name__ == '__main__':
    train()


