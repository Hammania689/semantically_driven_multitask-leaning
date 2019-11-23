# Vae Semantic Classification
Joint Reconstruction and Low-dimesional Latent Space Classification


# Getting Started

## Requirements 
WordNet's Perl Similarity Metric (see [wordNet.md](https://gitlab.com/usm-3d-search-lab/3d-scene-semantic-understanding/vae_semantic_classification/blob/master/wordNet.md))

PyTorch 1.1.0 (* May require nightly build... Pending on ongoing development) 

## **It is highly recommended that you use our Docker Image.**
### Build a Docker container
To abstract environment issues away Docker and it's gpu equivalent Nvidia Docker is used.

Download the Docker image: `docker pull chapchaebytes/semantic_mtl`

Build a Docker container with the required environment, GPU support and access to all of our files. **This only needs to be ran once if a container does not already exist**.
`(sudo) nvidia-docker run -it -v {host path}:{docker custom path} -w {repeat custom path} --name "{insert name here}"   chapchaebytes/semantic_mtl bash`

*One or two libraries and packages will still need to be installed but is quite trivial*

### Relevant Docker Commands 
Replace `{name/id of container}` with `pix2pixTF` or whatever you have named your container

Once you've built the container and wish to continue working:

    (sudo) nvidia-docker start {name/id of container}
    (sudo) nvidia-docker attach {name/id of container}

Stop a Container: `(sudo) nvidia-docker stop {name/id of container}`

Show all containers: `(sudo) nvidia-docker container ls`

Permanetly Remove stopped containers: `(sudo) nvidia-docker container prune`

# Running the scripts
To run the script provide the following: `python vae_train.py --Lambda=.9  --epochs=50 --batch_size=32 --arch="vae_gn"`

Current `--arch` arguments are `"vae"` , `"vae_bn",` and ` "vae_gn"`.
Refer to `models.py`.


Other parameters: 
![](https://i.imgur.com/R0BKjIr.png)

The following sections are being revised and W.I.P
---
## VAE latent space tractablity as improving reconstruction results

Review and form deeper understanding:
1. [Disentangled VAE for Semi-supervised Learning][r1]
2. [Learning Structured Output Representation using Deep Conditional Generative Models][r2]
3. [A Tutorial on Information Maximizing Variational Autoencoders (InfoVAE)][w1]
4. [Conditional Variational Autoencoder: Intuition and Implementation][w2]

# TODO
![Imgur](https://i.imgur.com/Dpyemo2.png)
- [ ] Update Logs accordingly loss = wi * loss
- [ ] Loss function normalization.

   Dividing each loss by 1000 did not yield desired results.
   Reconstruction loss has been decimated while LSTD loss is still rather large.
   Two Ideas:
   1. Divide each class by 1000 before summation in LSTD loss function
   2. Implement standard euclidean vector normalization ![norm](https://latex.codecogs.com/png.latex?$$\frac{V}{|V|}$)
- [ ] Training for multiple gpus
- [X] Joint loss function ( Reconstruction + LSTD + Classifier)
- [X] ~~Implement two separate optimizers. See [here][optim]~~
- [X] Add a proper Val dataset 
- [X] Implement Testing and Reconstruction Result Visualization

# If time allows
- [ ] Parallel dataloading see [here][dataload]

# Helpful PyTorch Forum Post
[How to train the network with multiple branches][h1]

[Two optimizers for one model][h2]

# Model Graph (as of 3.12.19)
![Imgur](https://i.imgur.com/U4D95SO.png)

[r1]: https://arxiv.org/pdf/1709.05047.pdf
[r2]: https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models.pdf
[w2]: https://wiseodd.github.io/techblog/2016/12/17/conditional-vae/
[w1]: https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
[optim]: https://discuss.pytorch.org/t/two-optimizers-for-one-model/11085/9
[dataload]: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel 
[h1]: https://discuss.pytorch.org/t/how-to-train-the-network-with-multiple-branches/2152
[h2]: https://discuss.pytorch.org/t/two-optimizers-for-one-model/11085
