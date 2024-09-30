from transformers import VisionEncoderDecoderModel , ViTFeatureExtractor
from transformers import AutoTokenizer ,  GPT2Config , default_data_collator

def load_pretrained(config):
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return outputs

    AutoTokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens

    feature_extractor = ViTFeatureExtractor.from_pretrained(config.ENCODER)
    
    tokenizer = AutoTokenizer.from_pretrained(config.DECODER)
    tokenizer.pad_token = tokenizer.unk_token

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(config.ENCODER, config.DECODER)
    model.to(config.device)
    
    return feature_extractor, tokenizer, model

def compute_parameters(model):
    # Total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Total number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return total_params, trainable_params