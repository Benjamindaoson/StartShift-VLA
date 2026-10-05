# Script execution policy

`verify_archive.py` is the supported archive reproduction entry point. It works offline with the standard library.

All other scripts are historical research tools. Bootstrap/download scripts access the network, `run_m0.sh` regenerates splits and launches evaluations, training helpers can launch experiments, and environment checks expect an independently installed robotics stack. They are retained to document the implementation, **not run as part of archive validation**. Python matrix scripts likewise remain historical even when they offer a dry-run mode.
