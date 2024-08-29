import logging

import numpy as np

from core import logger


def warn_gradient_too_small(actual, expected):
    logger.warn_once(f"Gradient size is too small for weight."
                     f" Expected: {expected}, Actual: {actual}")


def debug_gradient_too_big():
    logger.debug_once("Gradient size is too big")


def adapt_gradient(gradient, weight):
    if np.isscalar(weight.get()) and np.isscalar(gradient):
        return gradient
    if np.isscalar(weight.get()):
        return np.sum(gradient)
    try:
        gradient = gradient.reshape(weight.get().shape)
    except ValueError:
        try:
            if gradient.size > weight.get().size:
                gradient = np.sum(gradient, axis=0)
                debug_gradient_too_big()
            else:
                warn_gradient_too_small(gradient.shape, weight.get().shape)
                gradient = np.tile(gradient, (weight.get().shape[1], 1))
            gradient = gradient.reshape(weight.get().shape)
        except Exception as e:
            logger.logging.error(
                f"GRADIENT SHAPE {gradient.shape}; "
                f"WEIGHT SHAPE {weight.get().shape}"
            )
            logger.logging.error(e)
    return gradient
