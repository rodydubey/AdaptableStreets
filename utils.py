from gym.spaces import Box, Discrete

def get_space_dims(space):
    if isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Box):
        return space.shape[0]
    else:
        raise ValueError("space must be Discrete, Box")