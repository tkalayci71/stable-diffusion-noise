## Python quick recap

(I consider myself a programmer but I've been learning Python since getting interested in Stable Diffusion a few months back. A document like this would be useful to me at the beginning, so I made it hoping it could be useful to someone else,)

####Define a list:

    things = ['apple','orange','banana']
    print(things) # prints ['apple', 'orange', 'banana']
    print(len(things)) # prints 3
    print(things[0]) # prints apple
    print(things[1]) # prints orange
    print(things[2]) # prints banana

####Define a list of numbers:

    data = [10,20,30,40,50]
    print(data) # prints [10, 20, 30, 40, 50]
    print(len(data)) # prints 5
    print(data[0]) # prints 10
    print(data[4]) # prints 50

    some_data = data[1:4] # create a sub list, from [1], not including [4]
    print(some_data) # prints [20, 30, 40]

####To work exclusively on numbers, use numpy library:

    import numpy as np
    data = [0,1,2,3,4,5,6,7,8,9]
    data = np.array(data) # convert list to numpy array
    print(data) # prints [0 1 2 3 4 5 6 7 8 9]

    data = data * 2 # multiply each number by 2
    print(data) # prints [ 0  2  4  6  8 10 12 14 16 18]

####Numpy works on CPU. Torch library can work also on GPU/cuda:

    import torch
    data = [1,2,3,4]
    data = torch.tensor(data) # convert list to torch tensor
    print(data) # prints tensor([1, 2, 3, 4])

    data = data.to('cuda') # move the data to GPU
    print(data) # prints tensor([1, 2, 3, 4], device='cuda:0')
    print(data.device) # prints cuda:0

    new_data = torch.tensor([10,20,30,40],device='cuda')
    data = data + new_data # both tensors must be on the same device
    print(data) # prints tensor([11, 22, 33, 44], device='cuda:0')

    data = data.to('cpu') # move data back to CPU

####View/access a tensor as multi-dimensional arrays:

    data = torch.tensor([0,1,2,3,4,5,6,7])
    print(data.shape) # prints torch.Size([8])
    print(data[7]) # prints tensor(7)

    data = data.view(2,4) # view the same data as 2 x 4
    print(data.shape) # prints torch.Size([2, 4])
    print(data) # prints tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    print(data[0,3]) # prints tensor(3)
    print(data[1,0]) # prints tensor(4)

####Tensor data types:

    data = torch.tensor([1, 2, 3]) # integer data
    print(data.dtype) # prints torch.int64

    data = torch.tensor([3.14, 9.99]) # decimal data
    print(data.dtype) # prints torch.float32
    print(data) # prints tensor([3.1400, 9.9900])

    data = data.to(dtype=torch.float16) # convert to half-precision (16-bit)
    print(data.dtype) # prints torch.float16
    print(data) # prints tensor([3.1406, 9.9922], dtype=torch.float16)

####All sorts of operations are available:

    data = torch.tensor([1, 64, 100])
    data = data ** 0.5 # take the square root
    print(data) # prints tensor([ 1.,  8., 10.])

    data = torch.tensor([15.0, 20.0, 25.0])
    mean = torch.mean(data) # calculate the average
    print(mean) # prints tensor(20.0)
    std = torch.std(data, unbiased=True) # calculate standard deviation
    print(std) # prints tensor(5.)
