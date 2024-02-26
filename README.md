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
Epoch  0  loss  6.811458238540104 correct 34
Epoch  10  loss  4.700840112409717 correct 41
Epoch  20  loss  4.673468363227236 correct 45
Epoch  30  loss  3.1889921726622075 correct 45
Epoch  40  loss  3.2590963822426318 correct 44
Epoch  50  loss  4.08308968130736 correct 46
Epoch  60  loss  1.221887596185749 correct 47
Epoch  70  loss  1.5170521538873862 correct 48
Epoch  80  loss  0.942196064111543 correct 49
Epoch  90  loss  2.386292297587213 correct 48
Epoch  100  loss  0.9528669999902619 correct 50
Epoch  110  loss  1.368434353226853 correct 50
Epoch  120  loss  2.00104994968317 correct 49
```