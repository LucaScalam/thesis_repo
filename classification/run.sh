#!/bin/bash
      

# (
# python -u train.py -lr 5e-5 -e 300 -bs 16 -nn 64 -swa_opt 0 -idx 109 -nmod 20 -pmod 0 -td2 1 -coef 1 >train.py.log 2>&1
# python -u train.py -lr 1e-4 -e 300 -bs 16 -nn 64 -swa_opt 0 -idx 110 -nmod 20 -pmod 0 -td2 1 -coef 1 >train.py.log 2>&1
# python -u train.py -lr 5e-4 -e 300 -bs 16 -nn 64 -swa_opt 0 -idx 111 -nmod 20 -pmod 0 -td2 1 -coef 1 >train.py.log 2>&1
# ) &



# (
# python -u train_data_7.py > train_data_7.py.log 2>&1
# ) &


(
# python -u train_by_camp.py -lr 5e-5 -e 300 -bs 16 -nn 64 -swa_opt 0 -idx 120 -nmod 20 -pmod 0 -cmp 1819 -coef 1 >train_by_camp.py.log 2>&1
# python -u train_by_camp.py -lr 1e-4 -e 300 -bs 16 -nn 64 -swa_opt 0 -idx 121 -nmod 20 -pmod 0 -cmp 1819 -coef 1 >train_by_camp.py.log 2>&1
# python -u train_by_camp.py -lr 5e-5 -e 300 -bs 16 -nn 64 -swa_opt 0 -idx 122 -nmod 20 -pmod 0 -cmp 1920 -coef 1 >train_by_camp.py.log 2>&1
python -u train_by_camp.py -lr 1e-4 -e 300 -bs 16 -nn 64 -swa_opt 0 -idx 123 -nmod 20 -pmod 0 -cmp 1920 -coef 1 >train_by_camp.py.log 2>&1
) &

