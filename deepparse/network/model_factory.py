from . import FastTextSeq2SeqModel, BPEmbSeq2SeqModel


class ModelFactory:
    def create(
        self,
        model_type,
        cache_dir,
        device,
        output_size=9,
        verbose=True,
        path_to_retrained_model=None,
        attention_mechanism=False,
        **seq2seq_kwargs,
    ):

        if model_type == "fasttext":
            model = FastTextSeq2SeqModel(
                cache_dir=cache_dir,
                device=device,
                output_size=output_size,
                verbose=verbose,
                path_to_retrained_model=path_to_retrained_model,
                attention_mechanism=attention_mechanism,
                **seq2seq_kwargs,
            )

        elif model_type == "bpemb":
            model = BPEmbSeq2SeqModel(
                cache_dir=cache_dir,
                device=device,
                output_size=output_size,
                verbose=verbose,
                path_to_retrained_model=path_to_retrained_model,
                attention_mechanism=attention_mechanism,
                **seq2seq_kwargs,
            )
        else:
            raise NotImplementedError(
                f"""
                    There is no {model_type} network implemented. model_type should be either fasttext or bpemb
            """
            )

        return model
