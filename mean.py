def get_mean_std(value_scale):
    """
    Returns the mean and standard deviation of the given value scale for UCF-Crime.
    """

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    mean = [x * value_scale for x in mean]
    std = [x * value_scale for x in std]

    return mean, std
