from dotenv import load_dotenv

load_dotenv()

from keys import GeneralKeys

if GeneralKeys.RUNNER == "pre":
    from preprocessor.main import run

    run()
    exit()

if GeneralKeys.RUNNER == "train":
    from trainer.main import run

    run()
    exit()
