# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py

## Training Log
```
time python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
Epoch  0  loss  6.894071674572498 correct 40
Epoch  10  loss  4.479906507114382 correct 38
Epoch  20  loss  5.489764512868637 correct 39
Epoch  30  loss  5.27971586019496 correct 38
Epoch  40  loss  4.957002763946705 correct 42
Epoch  50  loss  4.753122145314738 correct 40
Epoch  60  loss  2.975124134769348 correct 44
Epoch  70  loss  3.734314672737835 correct 46
Epoch  80  loss  2.5898520854808575 correct 40
Epoch  90  loss  1.9332013222772417 correct 40
Epoch  100  loss  3.0839475523934934 correct 48
Epoch  110  loss  1.8581111276817315 correct 45
Epoch  120  loss  2.2793813834727987 correct 49
Epoch  130  loss  1.4926680681119484 correct 47
Epoch  140  loss  2.978261059264122 correct 50
Epoch  150  loss  1.2972488940489506 correct 48
Epoch  160  loss  3.2421291574956896 correct 48
Epoch  170  loss  1.2698212977400605 correct 50
Epoch  180  loss  1.4174222306761006 correct 50
Epoch  190  loss  0.45538721103656116 correct 49
Epoch  200  loss  0.7604060100277807 correct 50
Epoch  210  loss  1.0755531186439335 correct 50
Epoch  220  loss  0.4765162412329839 correct 50
Epoch  230  loss  0.21092762425878536 correct 50
Epoch  240  loss  0.9250252633506774 correct 50
Epoch  250  loss  0.4188434095719766 correct 50
Epoch  260  loss  0.42451244035065777 correct 50
Epoch  270  loss  1.0941340315738532 correct 50
Epoch  280  loss  2.6491632125447664 correct 48
Epoch  290  loss  1.1586046658019156 correct 50
Epoch  300  loss  0.7055136717822652 correct 50
Epoch  310  loss  0.8684926870906584 correct 50
Epoch  320  loss  0.6302049864121438 correct 50
Epoch  330  loss  0.6774214529784501 correct 50
Epoch  340  loss  0.38212173762427576 correct 50
Epoch  350  loss  0.7551756499157077 correct 50
Epoch  360  loss  0.7452086235847163 correct 50
Epoch  370  loss  0.2784807467801958 correct 50
Epoch  380  loss  0.17363169840100232 correct 50
Epoch  390  loss  0.8835791622114715 correct 50
Epoch  400  loss  0.47838284986418267 correct 50
Epoch  410  loss  0.24314054344049438 correct 50
Epoch  420  loss  0.4684875946049989 correct 50
Epoch  430  loss  0.19839144535687042 correct 50
Epoch  440  loss  0.19986038046543947 correct 50
Epoch  450  loss  0.2008121427402053 correct 50
Epoch  460  loss  0.3128638310014935 correct 50
Epoch  470  loss  0.3131961813608253 correct 50
Epoch  480  loss  0.16552736531227552 correct 50
Epoch  490  loss  0.42167788424986474 correct 50

real    3m41,032s
user    3m41,991s
sys     0m4,194s
```

```
time NUMBA_NUM_THREADS=16 python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
Epoch  0  loss  7.1445329235079456 correct 27
Epoch  10  loss  5.070881244737897 correct 36
Epoch  20  loss  5.0525709147550835 correct 46
Epoch  30  loss  4.203749772865221 correct 47
Epoch  40  loss  4.463384019983159 correct 43
Epoch  50  loss  2.0477194676040815 correct 49
Epoch  60  loss  3.4859661540614404 correct 47
Epoch  70  loss  2.789091267493724 correct 49
Epoch  80  loss  2.4123385779055377 correct 49
Epoch  90  loss  1.0138189151519674 correct 47
Epoch  100  loss  2.2969209956723646 correct 49
Epoch  110  loss  1.6544494233014235 correct 50
Epoch  120  loss  0.9648841854365396 correct 49
Epoch  130  loss  1.711884801630701 correct 48
Epoch  140  loss  2.15656484943929 correct 49
Epoch  150  loss  1.140002678610303 correct 50
Epoch  160  loss  0.1789329652043404 correct 50
Epoch  170  loss  0.7790420692307796 correct 49
Epoch  180  loss  0.5266274040068695 correct 50
Epoch  190  loss  0.45833374864229254 correct 50
Epoch  200  loss  1.263614420803114 correct 50
Epoch  210  loss  0.2857134609883469 correct 50
Epoch  220  loss  1.0880388755658856 correct 50
Epoch  230  loss  0.7016998913619973 correct 50
Epoch  240  loss  1.231804751288724 correct 50
Epoch  250  loss  1.749901548814637 correct 50
Epoch  260  loss  1.8272155433553474 correct 49
Epoch  270  loss  0.40504406542520344 correct 50
Epoch  280  loss  0.39657074872273307 correct 50
Epoch  290  loss  0.2700707063073223 correct 50
Epoch  300  loss  0.5364814299680037 correct 50
Epoch  310  loss  0.7934083566855238 correct 50
Epoch  320  loss  0.4340752665035075 correct 50
Epoch  330  loss  0.31245449243049783 correct 50
Epoch  340  loss  0.4552533431919098 correct 50
Epoch  350  loss  0.40502211369113034 correct 50
Epoch  360  loss  0.6704987142615315 correct 50
Epoch  370  loss  0.29531329386124183 correct 50
Epoch  380  loss  0.31697104429612344 correct 50
Epoch  390  loss  0.5945318941396678 correct 50
Epoch  400  loss  0.9352353280837774 correct 50
Epoch  410  loss  0.27678179294434657 correct 50
Epoch  420  loss  0.20413571224442825 correct 50
Epoch  430  loss  0.5884956136442041 correct 50
Epoch  440  loss  0.05978471236678132 correct 50
Epoch  450  loss  0.1268089083478271 correct 50
Epoch  460  loss  0.21142763437628376 correct 50
Epoch  470  loss  0.21475789415971083 correct 50
Epoch  480  loss  0.3807514163117207 correct 50
Epoch  490  loss  0.17113672683294162 correct 50

real    29m51,495s
user    32m29,932s
sys     19m29,064s
```