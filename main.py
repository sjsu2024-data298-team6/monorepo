from keys import GeneralKeys

if GeneralKeys.RUNNER == "pre":
    from preprocessor.preprocessor import run

    run()
    exit()

if GeneralKeys.RUNNER == "train":
    from trainer.trainer import run

    run()
    exit()
