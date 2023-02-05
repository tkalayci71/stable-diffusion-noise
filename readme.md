## Noise in Stable Diffusion, from a programmer's perspective

We've all seen images like this, where more noise is added to the image at each step:
![image](add_noise.png)

but how would one actually implement this? Below are my findings.

## The noising equation

This cryptic notation

![image](noise_equation.png)

is not helpful at all... not to me anyway

---

If Stable diffusion was trained with 1000 noising steps, I would imagine, naively,  a loop like:

    image = get_image()
    for step in range(1000):
        noise = generate_noise()
        image = image + noise

However, they forget to mention that noise is not just added, it's mixed, so it's more like:

        image = image * weight_image + noise * weight_noise

In reality, these weights are not constants, have different values at each step.
How to calculate them? Examining lms scheduler reveals this precalculation:

    num_train_timesteps = 1000
    beta_start = 0.00085
    beta_end = 0.012
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2

It turns out, the weights are calculated as:

        weight_image = (1-betas[step])**0.5
        weight_noise = betas[step]**0.5

which is what the equation was telling... 

## The noise

if you generate 100 random numbers, like:

    r = torch.rand(100)
    
the numbers you get will be between 0 and 1, and all values will be equally probable.

However, stable diffusion uses 'Gaussian noise', which you can generate by:

    r = torch.randn(100)

in this case, the numbers you get will be mostly between -3 and +3, and rarely out of this range. 
In fact, the probablities are known (68% between -1 and +1, etc.)

![image](gaussian.png)

If you calculate the average and standard deviation of this gaussian noise:

    import torch
    r = torch.randn(100)
    print( torch.mean(r) )
    print( torch.std(r,unbiased=True) )

you get a mean of ~ 0, which means negative and positive values are equally probable,
and standard deviation of ~ 1, which means 68% is between -1 and +1.

Interestingly, if you multiply the noise by k, standard deviation also gets multiplied by k (68% will be between -k and +k), while the mean remains at 0.  In other words, the bell curve will just get wider, and remain symmetrical around 0.

One last thing to know is, standard deviation is equal to the square root of 'variance'. Therefore, if you want to get noise with a specific variance, you multiply the normal noise by the square root of it. The betas above were variance values, that's the reason for the square roots (**0.5) you see.

In conclusion, generate_noise() function is just torch.randn() 
or rather, to make it have the same data structure as the image: torch.randn_like(image)

## The trick

The loop above is valid if you want to generate 1000 noisy versions of an image. However, note that, in the loop,  each noisy version is generated using the previous noisy version (similar to calculating compound interest). It turns out, using a math trick, it is possible to get the noisy image at any step, without calculating all the previous noisy versions.

![image](trick.png)

What this means is, to directly get the noisy image at any step, you can use:

    noisy_image_at_step = original_image * image_weight_acp  + noise * noise_weight_acp

again, this is precalculated in lms scheduler:

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

then:

    image_weight_acp = alphas_cumprod[step] ** 0.5
    noise_weight_acp = (1 - alphas_cumprod[step]) ** 0.5

if you check the noise weight value for the last step, it's 0.9977. It seems the beta start and end values were carefully selected to make the last noisy version almost pure noise.

## The VAE

The last part is, implementing get_image() function. You probably know that Stable Diffusion denoising doesn't use actual image pixels, but an encoded/compressed version of it. And the final output is decoded/decompressed into an actual image you can view. The model that does this is called the VAE (variable auto encoder).

After you have the VAE model loaded, you can use its encode/decode methods to convert between your PIL Image and the encoded images (called latent images).  As an example, if your image size is 512x512 with 3 channels (RGB), the latent image will be 64x64 with 4 channels.

I'm not including here the actual code to load PIL image and vae encoding/decoding, it's available elsewhere. Suffice it to say that some scaling is also applied involving a magic number 0.18215, I'm guessing to bring the value range close to -3 to +3 which is compatible with the gaussian noise.

---

Thanks for reading. Please let me know if you spot incorrect information here.

---

update: after simplifying lms scheduler by decreasing order to 1, I have obtained a simple diffusion loop which probably is the same as euler method. see https://github.com/tkalayci71/mini-diffusion for working implementation.
