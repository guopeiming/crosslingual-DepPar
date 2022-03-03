local bert_model = "xlm-roberta-base";
// local bert_model = "bert-base-multilingual-cased";
local token_indexers = {
  "tokens": {
    "type": "pretrained_transformer_mismatched",
    "model_name": bert_model,
  },
  "token_characters": {
    "type": "characters",
    "min_padding_length": 3
  },
};
local pgn_adapter_bert = {
  "type": "pgn_adapter_bert",
  "model_name": bert_model,
  "pgn_emb_num": "languages_labels",
  "pgn_emb_dim": 16,
};
local char_cnn = {
  "type": "character_encoding",
  "embedding": {
      "embedding_dim": 16,
      "vocab_namespace": "token_characters",
  },
  "encoder": {
      "type": "cnn",
      "embedding_dim": 16,
      "num_filters": 48,
      "ngram_filter_sizes": [3],
      "conv_layer_activation": "relu"
  }
};

{
  "random_seed": 2021,
  "numpy_seed": 11,
  "pytorch_seed": 15,

  "dataset_reader": {
    "type": "crosslingual_dep_parsing_reader",
    "token_indexers": token_indexers,
  },

  "train_data_path": "./data/data2/train/fine/st/gd.jsonl",
  "validation_data_path": "./data/en/dev/en0.jsonl",
  "test_data_path": "./data/gd/test/gd0.jsonl",
  "evaluate_on_test": true,
  "datasets_for_vocab_creation": ["train"],

  "model": {
    "type": "crosslingual_dep_parser",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": pgn_adapter_bert,
        "token_characters": char_cnn,
      },
    },
    "postags_embedding": {
      "embedding_dim": 48,
      "vocab_namespace": "pos",
    },
    "encoder": {
      "type": "stacked_bidirectional_lstm",
      "input_size": 768+48+48,
      "hidden_size": 512,
      "num_layers": 3,
      "recurrent_dropout_probability": 0.3,
      "use_highway": true
    },
    "use_mst_decoding_for_validation": true,
    "arc_representation_dim": 512,
    "tag_representation_dim": 128,
    "dropout": 0.3,
    "input_dropout": 0.3,
    // "initializer": {
    //   "regexes": [
    //     [".*tag_bilinear.*weight", {"type": "xavier_normal"}],
    //     [".*tag_bilinear.*bias", {"type": "uniform"}],
    //   ]
    // },
  },

  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 32,
      "sorting_keys": ["tokens"]
    }
  },

  "trainer": {
    "num_epochs": 35,
    "grad_norm": 5.0,
    "patience": 4,
    "validation_metric": "+LAS",
    "optimizer": {
      "type": "adamw",
      "lr": 1e-4,
      "weight_decay": 0.01,
    },
    "learning_rate_scheduler": {
      "type": "polynomial_decay",
      "warmup_steps": 500,
      "end_learning_rate": 1e-4,
    },
  }
}
