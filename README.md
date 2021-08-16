# AE
So far we have mostly used dense networks to learn
the features of the model spectra. Autoencoders, in particular
variational autoencoders (VAEs) offer advantages.
They have been used successfully to model galaxy spectra
in SDSS [10]. For each subspace, we train a variational
autoencoder, as shown in Figure 10, on intelligently engineered
features. The benefit of VAE over autoencoders is
that the VAE latent space is interpretable. As VAE maps
inputs into distributions instead of single values, its latent
space is continuous and is allowing interpolation between
similar inputs. Once trained, we can use the VAE as random
sampling engine in spectral space as well: we sample the
latent space and reconstruct spectrum from those parameters
using the decoder.
As real observational data start to come in, we will
modify the VAEs to learn the differences from the theoretical
models and incorporate the incoming new knowledge (noise,
and deviations of the real instrumental response from the
models) to improve the predictions of the spectra. This can
be accommodate by introducing additional latent variables
and start training those to represent these deviations.
