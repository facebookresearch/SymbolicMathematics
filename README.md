# Deep Learning for Symbolic Mathematics

PyTorch original implementation of [Deep Learning for Symbolic Mathematics](https://arxiv.org/abs/1912.01412) (ICLR 2020).

This repository contains code for:
- **Data generation**
    - Functions F with their derivatives f
    - Functions f with their primitives F
      - Forward (FWD)
      - Backward (BWD)
      - Integration by parts (IBP)
    - Ordinary differential equations with their solutions
      - First order (ODE1)
      - Second order (ODE2)
- **Training**
    - Half-precision (float16)
    - Multi-GPU
    - Multi-node
- **Evaluation**:
    - Greedy decoding
    - Beam search evaluation

We also provide:
- **Datasets**
    - Train / Valid / Test sets for all tasks considered in the paper
- **Trained models**
    - Models trained with different configurations of training data
- **Notebook**
    - An **[ipython notebook](https://github.com/facebookresearch/SymbolicMathematics/blob/master/beam_integration.ipynb)** with an interactive demo of the model on function integration




## Dependencies

- Python 3
- [NumPy](http://www.numpy.org/)
- [SymPy](https://www.sympy.org/)
- [PyTorch](http://pytorch.org/) (tested on version 1.3)
- [Apex](https://github.com/nvidia/apex#quick-start) (for fp16 training)

## Datasets and Trained Models

We provide datasets for each task considered in the paper:

| Dataset                       | #train     | Link                                                                            |
| ------------------------------|:----------:|:-------------------------------------------------------------------------------:|
| Integration (FWD)             |    45M     | [Link](https://dl.fbaipublicfiles.com/SymbolicMathematics/data/prim_fwd.tar.gz) |
| Integration (BWD)             |    88M     | [Link](https://dl.fbaipublicfiles.com/SymbolicMathematics/data/prim_bwd.tar.gz) |
| Integration (IBP)             |    23M     | [Link](https://dl.fbaipublicfiles.com/SymbolicMathematics/data/prim_ibp.tar.gz) |
| Differential equations (ODE1) |    65M     | [Link](https://dl.fbaipublicfiles.com/SymbolicMathematics/data/ode1.tar.gz)     |
| Differential equations (ODE2) |    32M     | [Link](https://dl.fbaipublicfiles.com/SymbolicMathematics/data/ode2.tar.gz)     |

We also provide models trained on the above datasets, for integration:

| Model training data | Accuracy (FWD) | Accuracy (BWD) | Accuracy (IBP) | Link                                                                              |
| --------------------|:--------------:|:--------------:|:--------------:|:---------------------------------------------------------------------------------:|
| FWD                 | 97.2%          | 16.1%          | 89.2%          | [Link](https://dl.fbaipublicfiles.com/SymbolicMathematics/models/fwd.pth)         |
| BWD                 | 31.6%          | 99.6%          | 60.0%          | [Link](https://dl.fbaipublicfiles.com/SymbolicMathematics/models/bwd.pth)         |
| IBP                 | 55.3%          | 85.5%          | 99.3%          | [Link](https://dl.fbaipublicfiles.com/SymbolicMathematics/models/ibp.pth)         |
| FWD + BWD           | 96.8%          | 99.6%          | 86.1%          | [Link](https://dl.fbaipublicfiles.com/SymbolicMathematics/models/fwd_bwd.pth)     |
| BWD + IBP           | 56.7%          | 99.5%          | 98.7%          | [Link](https://dl.fbaipublicfiles.com/SymbolicMathematics/models/ibp_bwd.pth)     |
| FWD + BWD + IBP     | 95.6%          | 99.5%          | 99.6%          | [Link](https://dl.fbaipublicfiles.com/SymbolicMathematics/models/fwd_bwd_ibp.pth) |

and for differential equations:

| Model training data | Accuracy (ODE1) | Accuracy (ODE2) | Link                                                                       |
| --------------------|:---------------:|:---------------:|:--------------------------------------------------------------------------:|
| ODE1                | 97.2%           | -               | [Link](https://dl.fbaipublicfiles.com/SymbolicMathematics/models/ode1.pth) |
| ODE2                | -               | 88.2%           | [Link](https://dl.fbaipublicfiles.com/SymbolicMathematics/models/ode2.pth) |

All accuracies above are given using a beam search of size 10. Note that these datasets and models slightly differ from the ones used in the paper.

## Data generation

If you want to use your own dataset / generator, it is possible to train a model by generating data on the fly.
However, the generation process can take a while, so we recommend to first generate data, and export it into a dataset that can be used for training. This can easily be done by setting `--export_data true`:

```bash
python main.py --export_data true

## main parameters
--batch_size 32
--cpu true
--exp_name prim_bwd_data
--num_workers 20               # number of processes
--tasks prim_bwd               # task (prim_fwd, prim_bwd, prim_ibp, ode1, ode2)
--env_base_seed -1             # generator seed (-1 for random seed)

## generator configuration
--n_variables 1                # number of variables (x, y, z)
--n_coefficients 0             # number of coefficients (a_0, a_1, a_2, ...)
--leaf_probs "0.75,0,0.25,0"   # leaf sampling probabilities
--max_ops 15                   # maximum number of operators (at generation, but can be much longer after derivation)
--max_int 5                    # max value of sampled integers
--positive true                # sign of sampled integers
--max_len 512                  # maximum length of generated equations

## considered operators, with (unnormalized) sampling probabilities
--operators "add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:1,acos:1,atan:1,sinh:1,cosh:1,tanh:1,asinh:1,acosh:1,atanh:1"

## other generations parameters can be found in `main.py` and `src/envs/char_sp.py`
```

Data will be exported in the prefix and infix formats to:
- `./dumped/prim_bwd_data/EXP_ID/data.prefix`
- `./dumped/prim_bwd_data/EXP_ID/data.infix`

`data.prefix` and `data.infix` are two parallel files containing the same number of lines, with the same equations written in prefix and infix representations respectively. In these files, each line contains an input (e.g. the function to integrate) and the associated output (e.g. an integral) separated by a tab. In practice, the model only operates on prefix data. The infix data is optional, but more human readable, and can be used for debugging purposes.

Note that some generators are very fast, such as `prim_bwd`, which only requires to generate a random function and to differentiate it. The others are significantly longer. For instance, the validity of differential equations is checked (symbolically and numerically) after generation, which can be expensive. In our case, we generated the data across a large number of CPUs to create a large training set. For reproducibility, we provide our training / validation / test datasets in the links above. Generators can be made faster by decreasing the timeout generation time in `char_sp.py`, but this may slightly reduce the set of equations that the generator can produce.

If you generate your own dataset, you will notice that the generator generates a lot of duplicates (which is inevitable if you parallelize the generation). In practice, we remove duplicates using:
```bash
cat ./dumped/prim_bwd_data/*/data.prefix \
| awk 'BEGIN{PROCINFO["sorted_in"]="@val_num_desc"}{c[$0]++}END{for (i in c) printf("%i|%s\n",c[i],i)}' \
> data.prefix.counts
```
The resulting format is the following:
```
count1|input1_prefix    output1_prefix
count2|input2_prefix    output2_prefix
...
```
Where the input and output are separated by a tab, and equations are sorted by counts. This is under this format that data has to be given to the model. The number of `counts` is not used by the model, but was not removed in case of potential curriculum learning. The last part consists in simply splitting the dataset into training / validation / test sets. This can be done with the `split_data.py` script:

```bash
# create a valid and a test set of 10k equations
python split_data.py data.prefix.counts 10000

# remove valid inputs that are in the train
mv data.prefix.counts.valid data.prefix.counts.valid.old
awk -F"[|\t]" 'NR==FNR { lines[$2]=1; next } !($2 in lines)' <(cat data.prefix.counts.train) data.prefix.counts.valid.old \
> data.prefix.counts.valid

# test test inputs that are in the train
mv data.prefix.counts.test data.prefix.counts.test.old
awk -F"[|\t]" 'NR==FNR { lines[$2]=1; next } !($2 in lines)' <(cat data.prefix.counts.train) data.prefix.counts.test.old \
> data.prefix.counts.test
```

## Training

To train a model, you first need data. You can either generate it using the scripts above, or download the data provided in this repository. For instance:

```bash
wget https://dl.fbaipublicfiles.com/SymbolicMathematics/data/prim_fwd.tar.gz
tar -xvf prim_fwd.tar.gz
```

Once you have a training / validation / test set, you can train using the following command:

```bash
python main.py

## main parameters
--exp_name first_train  # experiment name
--fp16 true --amp 2     # float16 training

## dataset location
--tasks "prim_fwd"                                                    # task
--reload_data "prim_fwd,prim_fwd.train,prim_fwd.valid,prim_fwd.test"  # data location
--reload_size 40000000                                                # training set size

## model parameters
--emb_dim 1024    # model dimension
--n_enc_layers 6  # encoder layers
--n_dec_layers 6  # decoder layers
--n_heads 8       # number of heads

## training parameters
--optimizer "adam,lr=0.0001"             # model optimizer
--batch_size 32                          # batch size
--epoch_size 300000                      # epoch size (number of equations per epoch)
--validation_metrics valid_prim_fwd_acc  # validation metric (when to save the model)
```

Additional training parameters can be found in `main.py`.

## Evaluation

During training, the accuracy on the validation set is measured at the end of each epoch. However, during training, we only compare the model generated output with the solution in the dataset, although the solution may not be unique. For instance, if the input is:

`y'' + y = 0`

and that the expected solution in the dataset is `a_0 * cos(x) + a_1 * sin(x)`, then if the model generates `a_0 * sin(x) + a_1 * cos(x)` the output will be considered invalid because it does not exactly match the one of the dataset.
To verify the model output, we plug it into the input equation to verify that this is a valid solution. However, manually verifying the model output can take a lot of time, so we only do this at the end of training, by setting `--beam_eval true`, and using the following command:

```bash
python main.py

## main parameters
--exp_name first_eval     # experiment name
--eval_only true          # evaluation mode (do not load the training set)
--reload_model "fwd.pth"  # model to reload and evaluate

## dataset location
--tasks "prim_fwd"                                                    # task
--reload_data "prim_fwd,prim_fwd.train,prim_fwd.valid,prim_fwd.test"  # data location

--emb_dim 1024    # model dimension
--n_enc_layers 6  # encoder layers
--n_dec_layers 6  # decoder layers
--n_heads 8       # number of heads

## evaluation parameters
--beam_eval true            # beam evaluation (with false, outputs are only compared with dataset solutions)
--beam_size 10              # beam size
--beam_length_penalty 1.0   # beam length penalty (1.0 corresponds to average of log-probs)
--beam_early_stopping 1     # beam early stopping
--eval_verbose 1            # export beam results (set to 2 to evaluate with beam even when greedy was successful)
--eval_verbose_print false  # print detailed evaluation results
```

Evaluation with beam can take some time, so we recommend to use not-too-large beams (10 is a good value).

## Frequently Asked Questions

### How can I run experiments on multiple GPUs?

This code supports both multi-GPU and multi-node training, and was tested with up to 128 GPUs. To run an experiment with multiple GPUs on a single machine, simply replace `python main.py` in the commands above with:

```bash
export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU main.py
```

The multi-node is automatically handled by SLURM.

### How can I use this code to train a model on a new task?

In `src/envs/char_sp.py` you will find several functions `gen_prim_fwd`, `gen_prim_bwd`, `gen_prim_ibp`, `gen_ode1`, `gen_ode2` responsible for the generation of the 5 tasks we considered, inside the environment class `CharSPEnvironment`. If you want to try a new task, you just need to add a new function `gen_NEW_TASK` to the environment class.

For all the tasks we considered, the input is composed of an equation with a function `y` which is the function to find. This procedure is compatible both with integration, and differential equations. For instance, in the case of integration, the input will be of the form `y' - F` where `F` is the function to integrate. In the case of differentiation, the input will be of the form `y - F'` where `F` is the function to differentiate. If the differential equation is `y'' + y = 0` the input will simply be `y'' + y`. At test time, the `y` function in the input is replaced by the output of the model which is considered valid if the input is evaluated to `0`. Based on the task you consider, you may need to update the evaluator in `evaluator.py` accordingly.

## References

[**Deep Learning for Symbolic Mathematics**](https://arxiv.org/abs/1912.01412) (ICLR 2020) - Guillaume Lample * and François Charton *

```
@article{lample2019deep,
  title={Deep learning for symbolic mathematics},
  author={Lample, Guillaume and Charton, Fran{\c{c}}ois},
  journal={arXiv preprint arXiv:1912.01412},
  year={2019}
}
```

## License

See the [LICENSE](LICENSE) file for more details.
