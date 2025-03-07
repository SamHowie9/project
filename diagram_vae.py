from pydrawnet.renderers import SeqRenderer, FreeformRenderer
from pydrawnet import layers, operations
import math
import matplotlib.pyplot as plt

SR = SeqRenderer()


SR.add_layer(layers.BlockLayer(256, 256))
SR.add_operation(operations.LinearOp())
SR.add_layer(layers.BlockLayer(128, 128))

SR.make_figure((16, 5))
SR.render(20)