# from tokenization import FullTokenizer
# import tensorflow as tf
# import os
#
# class BertWorker():
#     def __init__(self, args, graph_path, graph_config):
#         super().__init__()
#         self.max_seq_len = args.max_seq_len
#         self.mask_cls_sep = args.mask_cls_sep
#         self.daemon = True
#         self.prefetch_size = args.prefetch_size if self.device_id > 0 else None  # set to zero for CPU-worker
#         self.gpu_memory_fraction = args.gpu_memory_fraction
#         self.model_dir = args.model_dir
#         self.verbose = args.verbose
#         self.graph_path = graph_path
#         self.bert_config = graph_config
#         self.use_fp16 = args.fp16
#         self.show_tokens_to_client = args.show_tokens_to_client
#
#     def get_estimator(self, tf):
#         from tensorflow.python.estimator.estimator import Estimator
#         from tensorflow.python.estimator.run_config import RunConfig
#         from tensorflow.python.estimator.model_fn import EstimatorSpec
#
#         def model_fn(features, mode):
#             with tf.gfile.GFile(self.graph_path, 'rb') as f:
#                 graph_def = tf.GraphDef()
#                 graph_def.ParseFromString(f.read())
#
#             input_names = ['input_ids', 'input_mask', 'input_type_ids']
#
#             output = tf.import_graph_def(graph_def,
#                                          input_map={k + ':0': features[k] for k in input_names},
#                                          return_elements=['final_encodes:0'])
#
#             return EstimatorSpec(mode=mode, predictions={
#                 'encodes': output[0]
#             })
#
#         config = tf.ConfigProto(device_count={'GPU': 0 if self.device_id < 0 else 1})
#         config.gpu_options.allow_growth = True
#         config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
#         config.log_device_placement = False
#         # session-wise XLA doesn't seem to work on tf 1.10
#         # if args.xla:
#         #     config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
#
#         return Estimator(model_fn=model_fn, config=RunConfig(session_config=config))
#
#     def run(self, msg):
#         self._run(msg)
#
#     def _run(self, msg):
#         estimator = self.get_estimator(tf)
#         return estimator.predict(self.input_fn_builder(msg), yield_single_examples=False)
#
#     def input_fn_builder(self, msg):
#         from extract_features import convert_lst_to_features
#
#         def gen():
#             tokenizer = FullTokenizer(vocab_file=os.path.join(self.model_dir, 'vocab.txt'))
#             # Windows does not support logger in MP environment, thus get a new logger
#             # inside the process for better compatibility
#             # logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), self.verbose)
#
#             # logger.info('ready and listening!')
#
#             is_tokenized = all(isinstance(el, list) for el in msg)
#             tmp_f = list(convert_lst_to_features(msg, self.max_seq_len,
#                                                  self.bert_config.max_position_embeddings,
#                                                  tokenizer,
#                                                  is_tokenized, self.mask_cls_sep))
#
#             tokens = []
#             if self.show_tokens_to_client:
#                 tokens = [f.tokens for f in tmp_f]
#             # logger.info('gen: ' + str(tmp_f))
#             yield {
#                 # 'client_id': client_id,
#                 'input_ids': [f.input_ids for f in tmp_f],
#                 'input_mask': [f.input_mask for f in tmp_f],
#                 'input_type_ids': [f.input_type_ids for f in tmp_f],
#                 'tokens': tokens
#             }
#
#         def input_fn():
#             return (tf.data.Dataset.from_generator(
#                 gen,
#                 output_types={'input_ids': tf.int32,
#                               'input_mask': tf.int32,
#                               'input_type_ids': tf.int32,
#                               'client_id': tf.string},
#                 output_shapes={
#                     'client_id': (),
#                     'input_ids': (None, None),
#                     'input_mask': (None, None),
#                     'input_type_ids': (None, None)}).prefetch(self.prefetch_size))
#
#         return input_fn