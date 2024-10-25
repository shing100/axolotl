"""
DPO strategies for exaone chat template
"""


def argilla(
    cfg,
    **kwargs,
):  # pylint: disable=possibly-unused-variable,unused-argument
    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"[|system|]{sample['system']}[|endofturn|]\n"
                f"[|user|]{sample['instruction']}\n[|assistant|]"
            )
        else:
            sample[
                "prompt"
            ] = f"[|user|]{sample['instruction']}\n[|assistant|]"
        sample["chosen"] = f"{sample['chosen_response']}[|endofturn|]"
        sample["rejected"] = f"{sample['rejected_response']}[|endofturn|]"
        return sample

    return transform_fn


def argilla_chat(
    cfg,
    **kwargs,
):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    for argilla/dpo-mix-7k conversations
    """

    def transform_fn(sample):
        sample[
            "prompt"
        ] = f"[|user|]{sample['chosen'][0]['content']}\n[|assistant|]"
        sample["chosen"] = f"{sample['chosen'][1]['content']}[|endofturn|]"
        sample["rejected"] = f"{sample['rejected'][1]['content']}[|endofturn|]"
        return sample

    return transform_fn


def icr(
    cfg,
    **kwargs,
):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    chatml transforms for datasets with system, input, chosen, rejected
    ex. https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs
    """

    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"[|system|]{sample['system']}[|endofturn|]\n"
                f"[|user|]{sample['input']}\n[|assistant|]"
            )
        else:
            sample[
                "prompt"
            ] = f"[|user|]{sample['input']}\n[|assistant|]"
        sample["chosen"] = f"{sample['chosen']}[|endofturn|]"
        sample["rejected"] = f"{sample['rejected']}[|endofturn|]"
        return sample

    return transform_fn


def intel(cfg, **kwargs):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    For Intel Orca DPO Pairs
    """

    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"[|system|]{sample['system']}[|endofturn|]\n"
                f"[|user|]{sample['question']}\n[|assistant|]"
            )
        else:
            sample[
                "prompt"
            ] = f"[|user|]{sample['question']}\n[|assistant|]"
        sample["chosen"] = f"{sample['chosen']}[|endofturn|]"
        sample["rejected"] = f"{sample['rejected']}[|endofturn|]"
        return sample

    return transform_fn


def prompt_pairs(
    cfg, **kwargs
):  # pylint: disable=possibly-unused-variable,unused-argument
    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"[|system|]{sample['system']}[|endofturn|]\n"
                f"[|user|]{sample['prompt']}\n[|assistant|]"
            )
        else:
            sample[
                "prompt"
            ] = f"[|user|]{sample['prompt']}\n[|assistant|]"
        sample["chosen"] = f"{sample['chosen']}[|endofturn|]"
        sample["rejected"] = f"{sample['rejected']}[|endofturn|]"
        return sample

    return transform_fn


def ultra(cfg, **kwargs):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    for ultrafeedback binarized conversations
    """

    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"[|system|]{sample['system']}[|endofturn|]\n"
                f"[|user|]{sample['prompt']}\n[|assistant|]"
            )
        else:
            sample[
                "prompt"
            ] = f"[|user|]{sample['prompt']}\n[|assistant|]"
        sample["chosen"] = f"{sample['chosen'][1]['content']}[|endofturn|]"
        sample["rejected"] = f"{sample['rejected'][1]['content']}[|endofturn|]"
        return sample

    return transform_fn
