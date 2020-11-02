#### Environment Setup

1. The code uses Python 3.8,  [Pytorch 1.6.0](https://pytorch.org/) (Note that PyTorch 1.6.0 requires CUDA 10.2, if you want to run on a GPU)
2. Install PyTorch: `pip install pytorch` or `conda install pytorch torchvision -c pytorch` (the latter is recommended if you use Anaconda)
3. Install Python dependencies: `pip install -r requirements.txt`

#### Train SPoSE model 

```
  python main.py
  
 --task (specify whether you'd like the model to perform an odd-one-out (i.e., 3AFC) or similarity (i.e., 2AFC) task)
 --folder (folder where to load your triplets from or where to automatically create them from word embeddings (text) or neural activations (visual))
 --results_dir (if you would like to specify a directory for your results, this is where you can do it; else './results/' will be created for you)
 --tripletize (if you have pre-trained embeddings for N items or objects, the code can automatically tripletize them for you)
 --beta (if you want your pre-trained embeddings to be tripletized probabilistically, you can specify a beta value to determine the softmax temperature)
 --learning_rate (learning rate to be used in optimizer)
 --lmbda_idx (lambda value determines l1-norm fraction to regularize loss; index to access specific lambda value from a pre-defined range over k values))
 --embed_dim (embedding dimensionality, i.e., output size of the neural network)
 --batch_size (batch size)
 --epochs (maximum number of epochs to optimize SPoSE model for)
 --window_size (window size to be used for checking convergence criterion with linear regression)
 --sampling_method (sampling method; if soft, then you can specify a fraction of your training data to be sampled from during each epoch; else full train set will be used)
 --p (fraction of train set to sample)
 --plot_dims (whether or not to plot the number of non-negative dimensions as a function of time)
 --device (CPU or CUDA)
 --rnd_seed (random seed)
```

Here is an example call:

```
python main.py --task odd_one_out --folder behavioral/ --results_dir ./results/ --learning_rate 0.001 --lmbda_idx 0 --embed_dim 90 --batch_size 100 --epochs 500 --window_size 20 --sampling_method soft --p 0.7 --plot_dims --device cuda:0 --rnd_seed 42
```

#### NOTES:

1. The script expects your triplets to be in the folder `./triplets/behavioral/` or `./triplets/text/`, dependent on the data you use. Note that the triplets are expected to be in the format `N x 3`, where N = number of trials (e.g., 100k) and 3 refers to the triplets, where `col_0` = anchor_1, `col_1` = anchor_2, `col_2` = odd one out. Triplet data must be split into train and test splits, and named `train_90.txt` and `test_10.txt` respectively. In case you would like to use some sort of text embeddings (e.g., sensvecs), simply put your `.csv` files into the folder `./text/`, and the script will automatically tripletize the word embeddings for you and move the triplet data into `./triplets/text/`.  

2. The script automatically saves the weight matrix `W` of the SPoSE model at each convergence checkpoint. 

3. The script plots train and test performances alongside each other for each lambda value. All plots can be found in `./plots/` after model convergence.

4. For a specified lambda value, you get a `.json` file where both the best test performance(s) and the corresponding epoch at `max` performance are stored. You find the file in the results folder.

5. With the `--plot_dims` flag (see above) you may tell the script to plot the number of non-negative dimensions (i.e., weights > 0.1) as a function of time after the model has converged. This is useful to qualitatively inspect changes in non-negative dimensions over training epochs. Again, plots can be found in `./plots/` after model convergence.