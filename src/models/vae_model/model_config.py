class ModelConfig:
    def __init__(self, model_config_dict):
        self.z_dim = model_config_dict["z_dim"]
        self.hidden_dims = model_config_dict["hidden_dims"]
        self.relu_slope = model_config_dict.get("relu_slope", None)
        self.dropout = model_config_dict.get("dropout", None)
        self.batch_norm = model_config_dict.get("batch_norm", False)
        self.apply_sigmoid = model_config_dict.get("apply_sigmoid", False)

class Hyperparameters:
    def __init__(self, hyperparameters):
        self.epochs = hyperparameters.get("epochs")
        self.batch_size = hyperparameters.get("batch_size")
        self.loss_function = hyperparameters.get("loss_function")
        self.beta = hyperparameters.get("beta", -1) # Beta of -1 (default) will results in standard literature kl-weight
        self.epsilon = hyperparameters.get("epsilon", 1e-8) # Term added to the denominator to improve numerical stability (1e-8 is the PyTorch default values)
        self.learning_rate = hyperparameters.get("learning_rate", 0.01)
        self.weight_decay = hyperparameters.get("weight_decay", 0)

class LinearVAEConfig(ModelConfig):
    def __init__(self, model_config, hyperparameters):
        self.model_config = ModelConfig(model_config)
        self.model_config.input_dim = model_config['input_dim']
        self.hyperparameters = Hyperparameters(hyperparameters)

class DoubleEncoderLinearVAEConfig(ModelConfig):
    def __init__(self, model_config, hyperparameters):
        self.model_config = ModelConfig(model_config)
        self.model_config.input_dim = model_config['input_dim']
        self.model_config.decoder_two_dim = model_config['decoder_two_dim']
        self.hyperparameters = Hyperparameters(hyperparameters)

class DoubleDecoderCNNConfig(ModelConfig):
    def __init__(self, model_config, hyperparameters):
        self.model_config = ModelConfig(model_config)
        self.model_config.input_shape = model_config['input_shape']
        self.model_config.decoder_one_hidden_dims = model_config['decoder_one_hidden_dims']
        self.model_config.decoder_one_input_shape = model_config['decoder_one_input_shape']
        self.model_config.decoder_two_hidden_dims = model_config['decoder_two_hidden_dims']
        self.model_config.decoder_two_input_shape = model_config['decoder_two_input_shape']
        self.hyperparameters = Hyperparameters(hyperparameters)

class CNNVAEConfig(ModelConfig):
    def __init__(self, model_config, hyperparameters):
        self.model_config = ModelConfig(model_config)
        self.model_config.input_shape = model_config['input_shape']
        self.hyperparameters = Hyperparameters(hyperparameters)
