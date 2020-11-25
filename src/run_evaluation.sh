#!/bin/bash

# ===================================================================
# 1. HLSQG + sequential decoding
# ===================================================================
python eval-hlsqg.py --model_path hlsqg-p73k-tiny-out --logging_dir hlsqg-p73k-tiny-log
python eval-hlsqg.py --model_path hlsqg-p73k-mini-out --logging_dir hlsqg-p73k-mini-log
python eval-hlsqg.py --model_path hlsqg-p73k-small-out --logging_dir hlsqg-p73k-small-log
python eval-hlsqg.py --model_path hlsqg-p73k-medium-out --logging_dir hlsqg-p73k-medium-log
python eval-hlsqg.py --model_path hlsqg-p73k-base-out --logging_dir hlsqg-p73k-base-log

# ===================================================================
# 2. u-PMLM + sequential decoding + sample question token length
# ===================================================================
python eval-uPMLM.py --model_path uPMLM-p73k-tiny-out/ --logging_dir uPMLM-p73k-tiny-log  --sample_tok_len
python eval-uPMLM.py --model_path uPMLM-p73k-mini-out/ --logging_dir uPMLM-p73k-mini-log  --sample_tok_len
python eval-uPMLM.py --model_path uPMLM-p73k-small-out/ --logging_dir uPMLM-p73k-small-log  --sample_tok_len
python eval-uPMLM.py --model_path uPMLM-p73k-medium-out/ --logging_dir uPMLM-p73k-medium-log  --sample_tok_len
python eval-uPMLM.py --model_path uPMLM-p73k-base-out --logging_dir uPMLM-p73k-base-log  --sample_tok_len


# ===================================================================
# 3. u-PMLM + random decoding + sample question token length
# ===================================================================
python eval-uPMLM.py --model_path uPMLM-p73k-tiny-out/ --logging_dir uPMLM-p73k-tiny-log  --random_gen_order --sample_tok_len
python eval-uPMLM.py --model_path uPMLM-p73k-mini-out/ --logging_dir uPMLM-p73k-mini-log  --random_gen_order --sample_tok_len
python eval-uPMLM.py --model_path uPMLM-p73k-small-out/ --logging_dir uPMLM-p73k-small-log  --random_gen_order --sample_tok_len
python eval-uPMLM.py --model_path uPMLM-p73k-medium-out/ --logging_dir uPMLM-p73k-medium-log  --random_gen_order --sample_tok_len
python eval-uPMLM.py --model_path uPMLM-p73k-base-out --logging_dir uPMLM-p73k-base-log  --random_gen_order --sample_tok_len

# ===================================================================
# 4. u-PMLM + sequential decoding + given true question token length
# ===================================================================
python eval-uPMLM.py --model_path uPMLM-p73k-tiny-out/ --logging_dir uPMLM-p73k-tiny-log
python eval-uPMLM.py --model_path uPMLM-p73k-mini-out/ --logging_dir uPMLM-p73k-mini-log
python eval-uPMLM.py --model_path uPMLM-p73k-small-out/ --logging_dir uPMLM-p73k-small-log
python eval-uPMLM.py --model_path uPMLM-p73k-medium-out/ --logging_dir uPMLM-p73k-medium-log
python eval-uPMLM.py --model_path uPMLM-p73k-base-out --logging_dir uPMLM-p73k-base-log

# ===================================================================
# 5. u-PMLM + random decoding + given true question token length
# ===================================================================
python eval-uPMLM.py --model_path uPMLM-p73k-tiny-out/ --logging_dir uPMLM-p73k-tiny-log  --random_gen_order
python eval-uPMLM.py --model_path uPMLM-p73k-mini-out/ --logging_dir uPMLM-p73k-mini-log  --random_gen_order
python eval-uPMLM.py --model_path uPMLM-p73k-small-out/ --logging_dir uPMLM-p73k-small-log  --random_gen_order
python eval-uPMLM.py --model_path uPMLM-p73k-medium-out/ --logging_dir uPMLM-p73k-medium-log  --random_gen_order
python eval-uPMLM.py --model_path uPMLM-p73k-base-out --logging_dir uPMLM-p73k-base-log  --random_gen_order
