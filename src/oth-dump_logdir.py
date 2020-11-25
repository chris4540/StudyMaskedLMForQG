"""
Helper script to merge tensorboard log folder to json file and text file

Usage
-------
python oth-read_logdir.py --logdir <log_dir>


Notes
--------
Read the log directory by cmd line:
tensorboard --logdir <log_dir>
tensorboard --inspect --event_file=<event_file>
tensorboard --inspect --event_file=<event_file> --tag <tag>


Reference
---------------
tensorboard/backend/event_processing/event_accumulator.py
tensorflow_makefile/tensorflow/tensorboard/scripts/serialize_tensorboard.py
https://git.fh-muenster.de/dl337788/master-thesis/-/blob/9f11fb077b664d6176d6f4e2ba0e5f06a52dea4d/project/source/evaluation.py
"""

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json
from pathlib import Path
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir",
                        required=True,
                        help="the logdir to pass to the TensorBoard backend; "
                        "data will be read from this logdir for serialization."
                        )
    parser.add_argument("--target",
                        default=None,
                        help="The directoy where serialized data will be written"
                        )
    parser.add_argument("--overwrite", default=False,
                        help="Whether to remove and overwrite TARGET if it already exists.")

    args = parser.parse_args()
    logdir = args.logdir
    target = args.target
    if target is None:
        target = Path(logdir) / "dumps"
    print("target, where serialized data will be written: ", target)
    target.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # EventAccumulator
    # -----------------------
    ea = EventAccumulator(logdir)
    ea.Reload()

    # -----------------------
    # scalar results
    # -----------------------
    scalar_results = dict()
    for k in ea.scalars.Keys():
        scalar_events = [dict(e._asdict()) for e in ea.Scalars(k)]
        scalar_results[k] = scalar_events
    with open(target / "scalar_events.json", "w") as f:
        json.dump(scalar_results, f, indent=1)

    # -------------------------
    # Generated text
    # -------------------------
    for p in ["eval", "pred"]:
        try:
            text_gens = ea.Tensors(f"{p}_text_gen/text_summary")
        except KeyError:
            # just skip the text summary dump if cannot find it out
            continue

        for text in text_gens:
            step = text.step
            outfile = target / f"{p}_text_gen-{step}.txt"
            with open(outfile, "w", encoding="utf-8") as f:
                f.write(str(text.tensor_proto.string_val[0].decode()))
