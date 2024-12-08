from __future__ import absolute_import  #使用绝对导入方式，避免与本地模块冲突
from __future__ import division         #在整数除法中返回浮点结果
from __future__ import unicode_literals #使字符串默认是 Unicode
from __future__ import print_function   #启用 Python 3 的 print 函数语法，兼容 Python 2
                                        #提高兼容性
import torch
import numpy as np                      #用于数值计算和数组操作
import random                           #导入随机数生成模块，用于随机化操作
import os                               #导入操作系统接口模块，用于文件和路径操作
import json                             #用于处理 JSON 格式数据
import copy                             #深拷贝和浅拷贝操作
from metrics import (compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim)
                                        #导入计算指标和文本-视频相似度评估相关函数
                                        #compute_metrics：计算综合指标。
                                        #tensor_text_to_video_metrics：文本到视频匹配指标。
                                        #tensor_video_to_text_sim：视频到文本匹配相似度。
import time
import argparse                         #用于解析用户输入的命令行参数

from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer #用于分词操作
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE           #存储预训练模型
from modules.modeling import CLIP4IDC     #模型
from modules.optimization import BertAdam #优化器，用于模型训练
from modules.beam import Beam             #实现 Beam Search 解码

from util import parallel_apply, get_logger
                                        #parallel_apply：并行应用函数
                                        #get_logger：获取日志记录器，用于记录程序运行状态
from dataloaders.data_dataloaders import DATALOADER_DICT
                                        #数据加载器的字典，定义了不同任务的数据加载方式
from pycocotools.coco import COCO       #加载 COCO 数据集的标注
from pycocoevalcap.eval import COCOEvalCap  #评估生成的图像描述
from pycocoevalcap.eval import PTBTokenizer, Bleu, Meteor, Rouge, Cider
                                        #PTBTokenizer：标准分词器。
                                        #Bleu：计算 BLEU 分数。
                                        #Meteor：计算 METEOR 分数。
                                        #Rouge：计算 ROUGE 分数。
                                        #Cider：计算 CIDEr 分数。

torch.distributed.init_process_group(backend="nccl")
#初始化 PyTorch 分布式训练进程组，使用 NCCL（NVIDIA Collective Communications Library）作为通信后端。
#此代码用于分布式训练环境，NCCL 通常在多 GPU 设置下表现最佳。
global logger #记录重要事件和调试信息

class EvalCap(COCOEvalCap):
    def __init__(self,coco,cocoRes):
        super(EvalCap,self).__init__(coco,cocoRes)

    def evaluate(self):
        imgIds = self.params['Image_id']
        gts = {}  #图像的真实标注（Ground Truth, GT）
        res = {}  #模型生成的结果描述（Results）
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # set up scorers
        print('tokenization...')
        tokenizer = PTBTokenizer()  #初始化一个 PTBTokenizer 实例，用于对文本数据进行分词操作
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        # 调用 tokenize 方法对真实标注（gts）和模型生成的描述（res）进行分词

        # set up scorers
        print('setting up scorers...')
        scorers = [
            (Bleu(4),["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]), #计算蓝色分数（常用于评估机器翻译）
            (Meteor(),"METEOR"), #基于词义匹配的评估指标
            (Rouge(),"ROUGE_L"), #基于最长公共子序列（LCS）的评估指标
            (Cider(),"CIDEr"), #针对图像描述生成任务设计的评估指标
        ]

        # compute scorers
        for scorer,method in scorers:
            print('computing %s score...' % (scorer.method()))  # scorer.method() 返回评分器的名称
            score,scores = scorer.compute_score(gts,res)
            if type(method) == list:
                for sc,scs,m in zip(score,scores,method):
                    self.setEval(sc,m)  #记录全局评估分数 sc，对应方法名称 m
                    self.setImgToEvalImgs(scs,gts.keys(),m)  #记录每张图像的局部分数 scs
                    print("%s:%0.3f" % (m,sc))
                else:
                    self.setEval(score, method)
                    self.setImgToEvalImgs(scores, gts.keys(), m)
                    print("%s:%0.3f" % (m, sc))
            self.setEvalImgs() #汇总并保存所有图像的评估结果，用于后续输出或可视化

    def get_args(description='CLIP4IDC on Captioning Task'):
        parser = argparse.ArgumentParser(description=description) #用于解析命令行参数，这个对象的描述信息被设置为传入的 description 参数值
        parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
        parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
        parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
#布尔类型的标志，用于决定是否进行预训练、常规训练以及在开发集上进行评估
        parser.add_argument('--data_path', type=str, default='data/datatype', help='data file path')
        parser.add_argument('--features_path', type=str, default='data/datatype/images', help='feature path')
#数据文件路径和特征路径的参数
        parser.add_argument('--num_thread_reader', type=int, default=1, help='') #线程数
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate') #初始学习率
        parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit') #训练轮数上限
        parser.add_argument('--batch_size', type=int, default=256, help='batch size') #训练批次大小
        parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval') #评估批次大小
        parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay') #学习率衰减因子
        parser.add_argument('--n_display', type=int, default=100, help='Information display frequence') #信息显示频率
        parser.add_argument('--seed', type=int, default=42, help='random seed') #随机种子
        parser.add_argument('--max_words', type=int, default=20, help='') #文本最大单词数
        parser.add_argument('--margin', type=float, default=0.1, help='margin for loss') #损失函数的边界值
        parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample') #内部负样本的比率
        parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative') #加权值
        parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

        parser.add_argument("--output_dir", default=None, type=str, required=True,
                            help="The output directory where the model predictions and checkpoints will be written.")
        parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
        parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
        parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
        parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
        parser.add_argument("--do_lower_case", action='store_true',
                            help="Set this flag if you are using an uncased model.")
        parser.add_argument("--warmup_proportion", default=0.1, type=float,
                            help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

        parser.add_argument("--cache_dir", default="", type=str,
                            help="Where do you want to store the pre-trained models downloaded from s3")

        parser.add_argument('--fp16', action='store_true',
                            help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
        parser.add_argument('--fp16_opt_level', type=str, default='O1',
                            help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                 "See details at https://nvidia.github.io/apex/amp.html")

        parser.add_argument("--task_type", default="caption", type=str,
                            help="Point the task `retrieval` or `caption` to finetune.")
        parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")

        parser.add_argument("--world_size", default=0, type=int, help="distribted training")
        parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
        parser.add_argument("--rank", default=0, type=int, help="distribted training")
        parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
        parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
        parser.add_argument('--sampled_use_mil', action='store_true',
                            help="Whether MIL, has a high priority than use_mil.")

        parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
        parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
        parser.add_argument('--intra_num_hidden_layers', type=int, default=9, help="Layer NO. of intra module")
        parser.add_argument('--cross_num_hidden_layers', type=int, default=2, help="Layer NO. of cross.")

        parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
        parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                            help="linear projection of flattened patches.")

        parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")
        parser.add_argument("--gt_dir", default="gt", type=str,
                            help="The output directory where the model predictions and checkpoints will be written.")

        args = parser.parse_args()

        # Check paramenters
        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))
        if not args.do_train and not args.do_eval:
            raise ValueError("At least one of `do_train` or `do_eval` must be True.")

        args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

        return args

