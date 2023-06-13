from logger import get_logger
from preprocessor import Preprocessor
from config import get_configs, save_configs
from dataloader import get_datasets, Seq2SeqDataCollator
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    set_seed,
    Seq2SeqTrainingArguments
)
from transformers import BertTokenizer, BartForConditionalGeneration
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from transformers.trainer_utils import is_main_process
from transformers.training_args import ParallelMode
from transformers.tokenization_utils_base import AddedToken
from utils import *
from metrics import build_compute_metrics_fn
from torch.optim import AdamW

CSEP = "<c>"        # choices seperator

def main():
    # Get and save configs / logs
    configs = get_configs()
    model_args, data_args, training_args = configs
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(int(os.environ.get("LOCAL_RANK", -1))):
        logger = get_logger(training_args.logging_dir, training_args.logging_name+".log")
        save_configs(configs)

    # Set seed
    set_seed(training_args.seed)

    # Get tokenizer
    if "t5-pegasus" in model_args.model_name_or_path:
        tokenizer = T5PegasusTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
    elif "bart-base-chinese" in model_args.model_name_or_path:
        tokenizer = BertTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
    else: 
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )

    for token in ["DIGIT", CSEP]:
        tokenizer.add_tokens(AddedToken(token, single_word=True), special_tokens=False)

    if model_args.add_tokens:
        additional_tokens = ["<title>", "<choices>"]
        for token in additional_tokens:
            tokenizer.add_tokens(AddedToken(token, single_word=True), special_tokens=False)

    # Preprocess data
    preprocessor = Preprocessor(
        data_args.raw_data_dir, training_args.data_dir, data_args.schemas, 
        data_args.add_prompt, tokenizer.sep_token, CSEP)
    preprocessor.process()

    # Get dataloaders
    train_dataset, eval_dataset, test_dataset = get_datasets(data_args, training_args, tokenizer)

    # Define model, optimizer, metrics
    config = AutoConfig.from_pretrained(
        model_args.model_config if model_args.model_config else model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    if "t5-pegasus" in model_args.model_name_or_path:
        model = MT5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            from_tf=".ckpt" in model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir
        )
    elif "bart-base-chinese" in model_args.model_name_or_path:
        model = BartForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            from_tf=".ckpt" in model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir
        )
    else: 
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=".ckpt" in model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir
        )

    if model_args.add_tokens:
        model.resize_token_embeddings(len(tokenizer))

    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
    lr_schedular = None

    compute_metrics = (
        build_compute_metrics_fn(
            targets = data_args.targets,
            tokenizer = tokenizer, 
            output_dir = training_args.logging_dir, 
            data_dir = training_args.data_dir,
            cache_dir = training_args.cache_dir,
            extract_template = "default", 
            split = "valid",
            CSEP = CSEP,
            sort_choices = True
            ) if training_args.predict_with_generate else None
    )

    # Set num_beams for evaluation
    if data_args.eval_beams is None:
        data_args.eval_beams = model.config.num_beams

    # Define training arguments
    args = Seq2SeqTrainingArguments(
        do_train=training_args.do_train,
        do_eval=training_args.do_eval,
        num_train_epochs=training_args.num_train_epochs,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        warmup_ratio=training_args.warmup_ratio,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        gradient_checkpointing=training_args.gradient_checkpointing,
        run_name=training_args.run_name,
        output_dir=training_args.output_dir,
        save_total_limit=training_args.num_train_epochs,
        logging_dir=os.path.join(training_args.logging_dir, "tf_logs"),
        logging_steps=training_args.logging_steps,
        log_level="passive",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model=training_args.metric_for_best_model,
        seed=training_args.seed,
        predict_with_generate=training_args.predict_with_generate,
        resume_from_checkpoint=training_args.resume_from_checkpoint,
        generation_max_length=data_args.max_target_length, 
        generation_num_beams=data_args.eval_beams,
        report_to=training_args.report_to
    )
    logger.warning(
        "[Process rank: {}] [device: {}] [n_gpu: {}] [distributed training: {}] [16-bits training: {}]".format(
            args.local_rank, args.device, args.n_gpu,
            bool(args.parallel_mode == ParallelMode.DISTRIBUTED), args.fp16
        )
    )
    # Resize LR by #GPU
    if args.n_gpu > 0:                
        logger.warning(
            f"lr({args.learning_rate}) * #GPU({args.n_gpu}) = {args.learning_rate * args.n_gpu}")
        args.learning_rate = args.learning_rate * args.n_gpu

    # Set trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Seq2SeqDataCollator(tokenizer, data_args),
        compute_metrics=compute_metrics,
        optimizers=(optimizer, lr_schedular),
        tokenizer=tokenizer
    )

    all_metrics = {}
    # Training
    if training_args.do_train:
        logger.info("*** Train ***")

        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        metrics = train_result.metrics
        metrics["train_n_samples"] = data_args.n_train if data_args.n_train else len(train_dataset)
        metrics["best_checkpoint"] = int(trainer.state.best_model_checkpoint.split("-")[-1])
        metrics["best_metric"] = training_args.metric_for_best_model
        metrics["best_score"] = trainer.state.best_metric

        # This also saves the tokenizer
        trainer.save_model(os.path.join(training_args.output_dir, "best_model"))  

        if trainer.is_world_process_zero():  # Whether or not this process is the global main process
            handle_metrics("train", metrics, training_args.metrics_dir)
            all_metrics.update(metrics)

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(
                os.path.join(training_args.output_dir, "best_model", "trainer_state.json"))

            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            tokenizer.save_pretrained(os.path.join(training_args.output_dir, "best_model"))

    # Evaluation
    if training_args.do_eval:
        if model_args.load_checkpoint_from:
            logger.info(f"Loading model from {model_args.load_checkpoint_from}.")
            device = trainer.model.device
            trainer.model = trainer.model.from_pretrained(
                model_args.load_checkpoint_from, config=config).to(device)

        logger.info("*** Evaluate ***")

        eval_output = trainer.predict(
            test_dataset=eval_dataset,
            metric_key_prefix="eval", 
            max_length=data_args.max_target_length, 
            num_beams=data_args.eval_beams
        )
        metrics = eval_output.metrics
        metrics["eval_n_samples"] = data_args.n_val if data_args.n_val else len(eval_dataset)
        metrics["eval_loss"] = round(metrics["eval_loss"], 4)

        if trainer.is_world_process_zero():
            handle_metrics("eval", metrics, training_args.metrics_dir)
            all_metrics.update(metrics)

    if training_args.do_predict:
        if model_args.load_checkpoint_from:
            logger.info(f"Loading model from {model_args.load_checkpoint_from}).")
            device = trainer.model.device
            trainer.model = trainer.model.from_pretrained(model_args.load_checkpoint_from, config=config).to(device)

        logger.info("*** Predict / Test ***")

        compute_metrics = (
            build_compute_metrics_fn(
                targets = data_args.targets,
                tokenizer = tokenizer, 
                output_dir = training_args.logging_dir, 
                data_dir = training_args.data_dir,
                cache_dir = training_args.cache_dir,
                extract_template = "default", 
                split = "test",
                CSEP = CSEP,
                sort_choices = True
                ) if training_args.predict_with_generate else None
        )

        trainer.compute_metrics = compute_metrics

        test_output = trainer.predict(
            test_dataset=test_dataset,
            metric_key_prefix="test",
            max_length=data_args.max_target_length,
            num_beams=data_args.eval_beams,
        )
        metrics = test_output.metrics
        metrics["test_n_samples"] = data_args.n_test if data_args.n_test else len(test_dataset)
        trainer.log(metrics)

        if trainer.is_world_process_zero():
            metrics["test_loss"] = round(metrics["test_loss"], 4)
            handle_metrics("test", metrics, training_args.metrics_dir)
            all_metrics.update(metrics)

    if trainer.is_world_process_zero():
        save_json(all_metrics, os.path.join(training_args.metrics_dir, "results_all.json"))
        delete_checkpoints(training_args.output_dir)
        
    return all_metrics


if __name__ == "__main__":
    main()